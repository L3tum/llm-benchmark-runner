use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fs;

#[derive(Debug, Deserialize)]
pub struct Config {
    pub models: Vec<Model>,
    pub benchmarks: Vec<String>,
    pub benchmark: HashMap<String, serde_yaml::Value>,
    #[serde(default)]
    pub docker: DockerConfig,
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
    #[serde(default = "default_true")]
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
            mount_docker_socket: true,
        }
    }
}

fn default_true() -> bool {
    true
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
}

pub fn load_config(path: &str) -> Result<Config> {
    let content = fs::read_to_string(path).context("Failed to read config")?;
    serde_yaml::from_str(&content).context("Failed to parse config")
}

pub fn attach_docker_config(
    benchmark_config: serde_yaml::Value,
    docker: &DockerConfig,
) -> serde_yaml::Value {
    let docker_value = serde_yaml::to_value(docker).unwrap_or(serde_yaml::Value::Null);
    match benchmark_config {
        serde_yaml::Value::Mapping(mut map) => {
            map.insert(
                serde_yaml::Value::String("__docker".to_string()),
                docker_value,
            );
            serde_yaml::Value::Mapping(map)
        }
        serde_yaml::Value::Null => {
            let mut map = serde_yaml::Mapping::new();
            map.insert(
                serde_yaml::Value::String("__docker".to_string()),
                docker_value,
            );
            serde_yaml::Value::Mapping(map)
        }
        other => other,
    }
}
