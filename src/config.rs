use anyhow::{Context, Result};
use serde::Deserialize;
use std::collections::HashMap;
use std::fs;

#[derive(Debug, Deserialize)]
pub struct Config {
    pub models: Vec<Model>,
    pub benchmarks: Vec<String>,
    pub benchmark: HashMap<String, serde_yaml::Value>,
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
