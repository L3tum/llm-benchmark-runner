pub mod macro_expansion;

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fs;

#[derive(Debug, Deserialize)]
pub struct Config {
    pub models: Vec<Model>,
    pub benchmarks: Vec<String>,
    pub benchmark: HashMap<String, yaml_serde::Value>,
    #[serde(default)]
    pub docker: DockerConfig,
    /// Optional list of model comparisons to generate filtered reports for.
    #[serde(default)]
    pub comparisons: Vec<Comparison>,
}

#[derive(Debug, Deserialize)]
pub struct Comparison {
    /// Human-readable title for the comparison group.
    pub title: String,
    /// Display names of models to include in this comparison report.
    pub models: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DockerConfig {
    #[serde(default = "default_true")]
    pub enabled: bool,
    #[serde(default)]
    pub host_repo_path: Option<String>,
    #[serde(default = "default_docker_timeout_secs")]
    pub default_timeout_secs: u64,
    #[serde(default = "default_docker_images")]
    pub images: HashMap<String, String>,
    #[serde(default)]
    pub build_images: bool,
    #[serde(default = "default_max_workers")]
    pub max_workers: usize,
    #[serde(default = "default_docker_socket_path")]
    pub docker_socket_path: String,
    /// ⚠️ SECURITY WARNING: When `true`, the host Docker socket is mounted into
    /// benchmark containers. A container escape then grants **full host root
    /// access**. Only enable this in trusted environments (e.g. local dev where
    /// you own the host). Defaults to `false`.
    #[serde(default = "default_false")]
    pub mount_docker_socket: bool,
}

impl Default for DockerConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            host_repo_path: None,
            default_timeout_secs: default_docker_timeout_secs(),
            images: default_docker_images(),
            build_images: false,
            max_workers: default_max_workers(),
            docker_socket_path: default_docker_socket_path(),
            mount_docker_socket: false,
        }
    }
}

fn default_true() -> bool {
    true
}

fn default_false() -> bool {
    false
}

fn default_docker_timeout_secs() -> u64 {
    8
}

fn default_max_workers() -> usize {
    1
}

fn default_docker_socket_path() -> String {
    "/var/run/docker.sock".to_string()
}

fn default_docker_images() -> HashMap<String, String> {
    HashMap::from([
        ("python".to_string(), "python:3.12".to_string()),
        (
            "swebench_harness".to_string(),
            "llm-benchmark-runner/swebench-harness:latest".to_string(),
        ),
    ])
}

#[derive(Debug, Deserialize)]
pub struct Model {
    #[serde(alias = "model")]
    pub model_name: String,
    pub display_name: String,
    pub cmd: String,
    pub proxy: String,
    #[serde(default)]
    pub cmd_stop: Option<String>,
    /// Optional model-level parameters to merge into every request.
    /// These override any benchmark-level parameters.
    #[serde(default)]
    pub set_params: Option<HashMap<String, serde_json::Value>>,
    /// Optional minimum interval (ms) between API requests to this model.
    /// Throttles request rate to avoid provider rate limits (0/None = no limit).
    #[serde(default)]
    pub rate_limit_ms: Option<u64>,
}

pub fn load_config(path: &str) -> Result<Config> {
    let content = fs::read_to_string(path).context("Failed to read config")?;
    let value = macro_expansion::expand_config(&content)
        .context("Failed to expand config macros/variables")?;
    yaml_serde::from_value(value).context("Failed to parse config")
}

pub fn attach_docker_config(
    benchmark_config: yaml_serde::Value,
    docker: &DockerConfig,
) -> yaml_serde::Value {
    let docker_value = yaml_serde::to_value(docker).unwrap_or(yaml_serde::Value::Null);
    match benchmark_config {
        yaml_serde::Value::Mapping(mut map) => {
            map.insert(
                yaml_serde::Value::String("__docker".to_string()),
                docker_value,
            );
            yaml_serde::Value::Mapping(map)
        }
        yaml_serde::Value::Null => {
            let mut map = yaml_serde::Mapping::new();
            map.insert(
                yaml_serde::Value::String("__docker".to_string()),
                docker_value,
            );
            yaml_serde::Value::Mapping(map)
        }
        other => other,
    }
}

// Helper functions for extracting values from benchmark config
pub fn extract_usize(config: &yaml_serde::Value, key: &str) -> Option<usize> {
    if let yaml_serde::Value::Mapping(map) = config {
        if let Some(val) = map.get(yaml_serde::Value::String(key.to_string())) {
            return val.as_f64().map(|v| v as usize);
        }
    }
    None
}

pub fn extract_u64(config: &yaml_serde::Value, key: &str) -> Option<u64> {
    if let yaml_serde::Value::Mapping(map) = config {
        if let Some(val) = map.get(yaml_serde::Value::String(key.to_string())) {
            return val.as_f64().map(|v| v as u64);
        }
    }
    None
}

pub fn extract_string(config: &yaml_serde::Value, key: &str) -> Option<String> {
    if let yaml_serde::Value::Mapping(map) = config {
        if let Some(yaml_serde::Value::String(s)) =
            map.get(yaml_serde::Value::String(key.to_string()))
        {
            return Some(s.clone());
        }
    }
    None
}

pub fn extract_string_vec(config: &yaml_serde::Value, key: &str) -> Option<Vec<String>> {
    if let yaml_serde::Value::Mapping(map) = config {
        if let Some(val) = map.get(yaml_serde::Value::String(key.to_string())) {
            // Try as sequence first
            if let yaml_serde::Value::Sequence(arr) = val {
                return Some(
                    arr.iter()
                        .filter_map(|v| {
                            if let yaml_serde::Value::String(s) = v {
                                Some(s.clone())
                            } else {
                                None
                            }
                        })
                        .collect(),
                );
            }
            // Try as comma-separated string
            if let yaml_serde::Value::String(s) = val {
                if s.is_empty() {
                    return None;
                }
                return Some(s.split(',').map(|s| s.trim().to_string()).collect());
            }
        }
    }
    None
}

pub fn extract_bool(config: &yaml_serde::Value, key: &str) -> Option<bool> {
    if let yaml_serde::Value::Mapping(map) = config {
        if let Some(val) = map.get(yaml_serde::Value::String(key.to_string())) {
            // yaml_serde may represent booleans as strings (case-insensitive)
            if let yaml_serde::Value::Bool(b) = val {
                return Some(*b);
            }
            if let yaml_serde::Value::String(s) = val {
                let lower = s.trim().to_lowercase();
                return Some(matches!(lower.as_str(), "true" | "yes" | "1" | "on"));
            }
            // Try number
            if let Some(n) = val.as_f64() {
                return Some(n != 0.0);
            }
        }
    }
    None
}

pub fn extract_docker_config(config: &yaml_serde::Value) -> DockerConfig {
    if let yaml_serde::Value::Mapping(map) = config {
        if let Some(docker_val) = map.get(yaml_serde::Value::String("__docker".to_string())) {
            if let Ok(dc) = yaml_serde::from_value(docker_val.clone()) {
                return dc;
            }
        }
    }
    DockerConfig::default()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_config(pairs: Vec<(&str, serde_json::Value)>) -> yaml_serde::Value {
        let json: serde_json::Map<String, serde_json::Value> =
            pairs.into_iter().map(|(k, v)| (k.to_string(), v)).collect();
        yaml_serde::from_value(yaml_serde::to_value(json).unwrap()).unwrap()
    }

    #[test]
    fn extract_usize_from_number() {
        let cfg = make_config(vec![("n", serde_json::json!(42))]);
        assert_eq!(extract_usize(&cfg, "n"), Some(42));
    }

    #[test]
    fn extract_usize_missing_key() {
        let cfg = make_config(vec![]);
        assert_eq!(extract_usize(&cfg, "missing"), None);
    }

    #[test]
    fn extract_string_from_string() {
        let cfg = make_config(vec![("name", serde_json::json!("hello"))]);
        assert_eq!(extract_string(&cfg, "name"), Some("hello".to_string()));
    }

    #[test]
    fn extract_string_vec_from_sequence() {
        let cfg = make_config(vec![("tags", serde_json::json!(["a", "b"]))]);
        assert_eq!(
            extract_string_vec(&cfg, "tags"),
            Some(vec!["a".into(), "b".into()])
        );
    }

    #[test]
    fn extract_string_vec_from_csv() {
        let cfg = make_config(vec![("tags", serde_json::json!("x, y, z"))]);
        assert_eq!(
            extract_string_vec(&cfg, "tags"),
            Some(vec!["x".into(), "y".into(), "z".into()])
        );
    }

    #[test]
    fn extract_bool_true() {
        let cfg = make_config(vec![("on", serde_json::json!("true"))]);
        assert_eq!(extract_bool(&cfg, "on"), Some(true));
    }

    #[test]
    fn extract_bool_false() {
        let cfg = make_config(vec![("off", serde_json::json!("false"))]);
        assert_eq!(extract_bool(&cfg, "off"), Some(false));
    }

    #[test]
    fn extract_bool_native_yaml_bool() {
        // Unquoted YAML `true`/`false` deserialize to yaml_serde::Value::Bool,
        // not a string. This must be handled explicitly.
        let cfg = make_config(vec![("on", serde_json::json!(true))]);
        assert_eq!(extract_bool(&cfg, "on"), Some(true));
        let cfg = make_config(vec![("off", serde_json::json!(false))]);
        assert_eq!(extract_bool(&cfg, "off"), Some(false));
    }

    #[test]
    fn extract_bool_from_number() {
        let cfg = make_config(vec![("n", serde_json::json!(1))]);
        assert_eq!(extract_bool(&cfg, "n"), Some(true));
    }

    #[test]
    fn extract_bool_case_insensitive() {
        for s in ["True", "TRUE", "Yes", "YES", "1", "On"] {
            let cfg = make_config(vec![("on", serde_json::json!(s))]);
            assert_eq!(
                extract_bool(&cfg, "on"),
                Some(true),
                "expected true for {:?}",
                s
            );
        }
        for s in ["False", "FALSE", "No", "NO", "0", "Off"] {
            let cfg = make_config(vec![("off", serde_json::json!(s))]);
            assert_eq!(
                extract_bool(&cfg, "off"),
                Some(false),
                "expected false for {:?}",
                s
            );
        }
    }

    #[test]
    fn extract_u64_from_number() {
        let cfg = make_config(vec![("timeout", serde_json::json!(900))]);
        assert_eq!(extract_u64(&cfg, "timeout"), Some(900));
    }

    #[test]
    fn extract_docker_config_defaults() {
        let cfg = yaml_serde::Value::Null;
        let dc = extract_docker_config(&cfg);
        assert!(dc.enabled);
    }
}
