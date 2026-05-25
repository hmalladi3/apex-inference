//! YAML config schema and validation.
//!
//! See `docs/llds/config-and-reload.md` and `docs/specs/config-and-reload.md`.

use std::collections::HashSet;
use std::net::SocketAddr;
use std::path::{Path, PathBuf};

use serde::Deserialize;

use crate::error::ConfigError;

#[derive(Debug, Clone, Deserialize)]
pub struct Config {
    pub server: ServerConfig,
    #[serde(default)]
    pub observability: ObservabilityConfig,
    #[serde(default)]
    pub admission: AdmissionConfig,
    pub models: Vec<ModelConfig>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct ServerConfig {
    pub listen: String,
    #[serde(default = "default_request_timeout_secs")]
    pub request_timeout_secs: u64,
    #[serde(default = "default_shutdown_grace_secs")]
    pub shutdown_grace_secs: u64,
    #[serde(default = "default_max_request_bytes")]
    pub max_request_bytes: usize,
}

#[derive(Debug, Clone, Default, Deserialize)]
pub struct ObservabilityConfig {
    pub metrics_listen: Option<String>,
    pub otlp_endpoint: Option<String>,
    #[serde(default)]
    pub log_format: LogFormat,
    #[serde(default = "default_log_level")]
    pub log_level: String,
}

#[derive(Debug, Clone, Copy, Default, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum LogFormat {
    #[default]
    Json,
    Pretty,
}

#[derive(Debug, Clone, Default, Deserialize)]
pub struct AdmissionConfig {
    pub max_rss_bytes: Option<u64>,
    pub max_queue_depth: Option<usize>,
    #[serde(default = "default_rss_sample_interval_ms")]
    pub rss_sample_interval_ms: u64,
}

#[derive(Debug, Clone, Deserialize)]
pub struct ModelConfig {
    pub name: String,
    #[serde(default = "default_version")]
    pub version: String,
    pub path: PathBuf,
    pub kind: ModelKind,
    pub max_batch_size: usize,
    pub max_queue_delay_us: u64,
    pub intra_op_threads: usize,
    #[serde(default)]
    pub runtime: BridgeRuntimeKind,
    pub seq_len_buckets: Option<Vec<u32>>,
    #[serde(default)]
    pub requires_attention_mask: bool,
}

#[derive(Debug, Clone, Copy, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum ModelKind {
    FixedShape,
    Ragged,
}

#[derive(Debug, Clone, Copy, Default, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum BridgeRuntimeKind {
    #[default]
    Blocking,
    DedicatedThread,
}

impl Config {
    pub fn from_yaml_str(s: &str) -> Result<Self, ConfigError> {
        serde_yaml::from_str(s).map_err(|e| ConfigError::Parse(e.to_string()))
    }

    pub fn from_yaml_file(path: &Path) -> Result<Self, ConfigError> {
        let s = std::fs::read_to_string(path)
            .map_err(|e| ConfigError::Parse(format!("read {}: {e}", path.display())))?;
        Self::from_yaml_str(&s)
    }

    /// @spec CONFIG-VAL-001, CONFIG-VAL-002, CONFIG-VAL-003, CONFIG-VAL-004,
    ///       CONFIG-VAL-005, CONFIG-VAL-006, CONFIG-VAL-007, CONFIG-VAL-008,
    ///       CONFIG-VAL-009
    pub fn validate(&self) -> Result<(), ConfigError> {
        self.server
            .listen
            .parse::<SocketAddr>()
            .map_err(|e| ConfigError::Validation(format!("server.listen: {e}")))?;

        let mut seen: HashSet<(&str, &str)> = HashSet::new();
        for m in &self.models {
            if !seen.insert((m.name.as_str(), m.version.as_str())) {
                return Err(ConfigError::Validation(format!(
                    "duplicate (name, version): ({}, {})",
                    m.name, m.version
                )));
            }
            m.validate()?;
        }
        Ok(())
    }
}

impl ModelConfig {
    fn ident(&self) -> String {
        format!("{}/{}", self.name, self.version)
    }

    fn validate(&self) -> Result<(), ConfigError> {
        if !(1..=1024).contains(&self.max_batch_size) {
            return Err(ConfigError::Validation(format!(
                "{}: max_batch_size must be in [1, 1024], got {}",
                self.ident(),
                self.max_batch_size
            )));
        }
        if !(100..=1_000_000).contains(&self.max_queue_delay_us) {
            return Err(ConfigError::Validation(format!(
                "{}: max_queue_delay_us must be in [100, 1_000_000], got {}",
                self.ident(),
                self.max_queue_delay_us
            )));
        }
        match self.kind {
            ModelKind::FixedShape => {
                if self.seq_len_buckets.is_some() {
                    return Err(ConfigError::Validation(format!(
                        "{}: fixed_shape model must not declare seq_len_buckets",
                        self.ident()
                    )));
                }
            }
            ModelKind::Ragged => {
                let buckets = self.seq_len_buckets.as_ref().ok_or_else(|| {
                    ConfigError::Validation(format!(
                        "{}: ragged model must declare seq_len_buckets",
                        self.ident()
                    ))
                })?;
                if buckets.is_empty() {
                    return Err(ConfigError::Validation(format!(
                        "{}: seq_len_buckets must be non-empty",
                        self.ident()
                    )));
                }
                for w in buckets.windows(2) {
                    if w[0] >= w[1] {
                        return Err(ConfigError::Validation(format!(
                            "{}: seq_len_buckets must be strictly ascending",
                            self.ident()
                        )));
                    }
                }
                if let Some(&max) = buckets.iter().max()
                    && max > 65_536
                {
                    return Err(ConfigError::Validation(format!(
                        "{}: seq_len_buckets values must be ≤ 65536, got {}",
                        self.ident(),
                        max
                    )));
                }
            }
        }
        if !self.path.exists() {
            return Err(ConfigError::Validation(format!(
                "{}: model file does not exist: {}",
                self.ident(),
                self.path.display()
            )));
        }
        Ok(())
    }
}

fn default_request_timeout_secs() -> u64 {
    30
}
fn default_shutdown_grace_secs() -> u64 {
    30
}
fn default_max_request_bytes() -> usize {
    64 * 1024 * 1024
}
fn default_rss_sample_interval_ms() -> u64 {
    100
}
fn default_log_level() -> String {
    "info".to_string()
}
fn default_version() -> String {
    "1".to_string()
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    use tempfile::{NamedTempFile, TempDir};

    /// Build a config with one minimal fixed-shape model whose path is a
    /// real existing tempfile. Tests can mutate fields before validating.
    struct Fixture {
        config: Config,
        _model_file: NamedTempFile,
        _tmpdir: TempDir,
    }

    impl Fixture {
        fn new() -> Self {
            let tmpdir = tempfile::tempdir().unwrap();
            let model_file = NamedTempFile::new_in(tmpdir.path()).unwrap();
            let config = Config {
                server: ServerConfig {
                    listen: "0.0.0.0:9000".to_string(),
                    request_timeout_secs: 30,
                    shutdown_grace_secs: 30,
                    max_request_bytes: 64 * 1024 * 1024,
                },
                observability: ObservabilityConfig::default(),
                admission: AdmissionConfig::default(),
                models: vec![ModelConfig {
                    name: "m".to_string(),
                    version: "1".to_string(),
                    path: model_file.path().to_path_buf(),
                    kind: ModelKind::FixedShape,
                    max_batch_size: 32,
                    max_queue_delay_us: 4000,
                    intra_op_threads: 4,
                    runtime: BridgeRuntimeKind::Blocking,
                    seq_len_buckets: None,
                    requires_attention_mask: false,
                }],
            };
            Self {
                config,
                _model_file: model_file,
                _tmpdir: tmpdir,
            }
        }
    }

    fn assert_validation_err(result: Result<(), ConfigError>, needle: &str) {
        match result {
            Err(ConfigError::Validation(msg)) => {
                assert!(
                    msg.contains(needle),
                    "expected validation error containing {needle:?}, got: {msg}"
                );
            }
            other => panic!("expected validation error, got {other:?}"),
        }
    }

    /// Happy path.
    #[test]
    fn valid_minimal_config_passes_validation() {
        let f = Fixture::new();
        f.config.validate().expect("valid config should validate");
    }

    /// @spec CONFIG-VAL-001
    #[test]
    fn invalid_listen_address_fails() {
        let mut f = Fixture::new();
        f.config.server.listen = "not a socket addr".to_string();
        assert_validation_err(f.config.validate(), "server.listen");
    }

    /// @spec CONFIG-VAL-003
    #[test]
    fn zero_max_batch_size_fails() {
        let mut f = Fixture::new();
        f.config.models[0].max_batch_size = 0;
        assert_validation_err(f.config.validate(), "max_batch_size");
    }

    /// @spec CONFIG-VAL-003
    #[test]
    fn max_batch_size_over_limit_fails() {
        let mut f = Fixture::new();
        f.config.models[0].max_batch_size = 2048;
        assert_validation_err(f.config.validate(), "max_batch_size");
    }

    /// @spec CONFIG-VAL-004
    #[test]
    fn out_of_range_queue_delay_fails() {
        let mut f = Fixture::new();
        f.config.models[0].max_queue_delay_us = 50;
        assert_validation_err(f.config.validate(), "max_queue_delay_us");
    }

    /// @spec CONFIG-VAL-005
    #[test]
    fn ragged_with_empty_buckets_fails() {
        let mut f = Fixture::new();
        f.config.models[0].kind = ModelKind::Ragged;
        f.config.models[0].seq_len_buckets = Some(vec![]);
        assert_validation_err(f.config.validate(), "seq_len_buckets");
    }

    /// @spec CONFIG-VAL-005
    #[test]
    fn ragged_with_non_ascending_buckets_fails() {
        let mut f = Fixture::new();
        f.config.models[0].kind = ModelKind::Ragged;
        f.config.models[0].seq_len_buckets = Some(vec![128, 64, 256]);
        assert_validation_err(f.config.validate(), "ascending");
    }

    /// @spec CONFIG-VAL-006
    #[test]
    fn ragged_with_oversized_bucket_fails() {
        let mut f = Fixture::new();
        f.config.models[0].kind = ModelKind::Ragged;
        f.config.models[0].seq_len_buckets = Some(vec![64, 128, 100_000]);
        assert_validation_err(f.config.validate(), "65536");
    }

    /// @spec CONFIG-VAL-007
    #[test]
    fn fixed_shape_with_buckets_fails() {
        let mut f = Fixture::new();
        f.config.models[0].seq_len_buckets = Some(vec![64, 128]);
        assert_validation_err(f.config.validate(), "fixed_shape");
    }

    /// @spec CONFIG-VAL-008
    #[test]
    fn nonexistent_model_path_fails() {
        let mut f = Fixture::new();
        f.config.models[0].path = PathBuf::from("/this/does/not/exist.onnx");
        assert_validation_err(f.config.validate(), "does not exist");
    }

    /// @spec CONFIG-VAL-009
    #[test]
    fn duplicate_name_version_pair_fails() {
        let mut f = Fixture::new();
        let dup = ModelConfig {
            name: f.config.models[0].name.clone(),
            version: f.config.models[0].version.clone(),
            path: f.config.models[0].path.clone(),
            kind: ModelKind::FixedShape,
            max_batch_size: 32,
            max_queue_delay_us: 4000,
            intra_op_threads: 4,
            runtime: BridgeRuntimeKind::Blocking,
            seq_len_buckets: None,
            requires_attention_mask: false,
        };
        f.config.models.push(dup);
        assert_validation_err(f.config.validate(), "duplicate");
    }

    /// Same name with distinct versions is allowed.
    /// @spec CONFIG-VAL-009
    #[test]
    fn same_name_different_version_is_allowed() {
        let mut f = Fixture::new();
        let mut alt = ModelConfig {
            name: f.config.models[0].name.clone(),
            version: "2".to_string(),
            path: f.config.models[0].path.clone(),
            kind: ModelKind::FixedShape,
            max_batch_size: 32,
            max_queue_delay_us: 4000,
            intra_op_threads: 4,
            runtime: BridgeRuntimeKind::Blocking,
            seq_len_buckets: None,
            requires_attention_mask: false,
        };
        alt.version = "2".to_string();
        f.config.models.push(alt);
        f.config
            .validate()
            .expect("(name, v1) and (name, v2) should validate");
    }

    #[test]
    fn yaml_parse_roundtrip() {
        let tmpdir = tempfile::tempdir().unwrap();
        let model_file = NamedTempFile::new_in(tmpdir.path()).unwrap();
        let path = model_file.path().display();
        let yaml = format!(
            r#"
server:
  listen: "0.0.0.0:9000"

models:
  - name: m
    path: {path}
    kind: fixed_shape
    max_batch_size: 32
    max_queue_delay_us: 4000
    intra_op_threads: 4
"#
        );
        let cfg = Config::from_yaml_str(&yaml).expect("yaml should parse");
        cfg.validate().expect("parsed yaml should validate");
        assert_eq!(cfg.models[0].name, "m");
        assert_eq!(cfg.models[0].version, "1"); // default applied
        assert_eq!(cfg.models[0].kind, ModelKind::FixedShape);
    }

    #[test]
    fn yaml_parse_error_is_typed() {
        let bad = "this: is: not: valid\n  yaml";
        let err = Config::from_yaml_str(bad).expect_err("malformed yaml should fail");
        assert!(matches!(err, ConfigError::Parse(_)));
    }

    #[test]
    fn from_yaml_file_reads_disk() {
        let tmpdir = tempfile::tempdir().unwrap();
        let model_file = NamedTempFile::new_in(tmpdir.path()).unwrap();
        let path = model_file.path().display();
        let yaml = format!(
            r#"
server:
  listen: "127.0.0.1:9001"
models:
  - name: m
    path: {path}
    kind: fixed_shape
    max_batch_size: 8
    max_queue_delay_us: 1000
    intra_op_threads: 2
"#
        );
        let mut cfg_file = NamedTempFile::new_in(tmpdir.path()).unwrap();
        cfg_file.write_all(yaml.as_bytes()).unwrap();
        let cfg = Config::from_yaml_file(cfg_file.path()).expect("should parse");
        cfg.validate().expect("should validate");
        assert_eq!(cfg.server.listen, "127.0.0.1:9001");
    }
}
