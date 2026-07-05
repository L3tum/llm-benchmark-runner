use crate::config::Model;
use anyhow::Result;
use std::collections::HashMap;
use std::sync::OnceLock;

pub mod kld;
pub mod mmlu_pro;

/// Trait for all benchmarks.
pub trait Benchmark: Send + Sync {
    #[expect(dead_code)]
    fn name(&self) -> &str;
    fn pre_execute(&self, _config: &serde_yaml::Value) -> Result<()> {
        Ok(())
    }
    fn execute(&self, model: &Model, config: &serde_yaml::Value) -> Result<serde_json::Value>;
    fn post_execute(
        &self,
        _model_results: &HashMap<String, serde_json::Value>,
    ) -> Result<serde_json::Value> {
        Ok(serde_json::Value::Null)
    }
}

fn registry() -> &'static HashMap<String, Box<dyn Benchmark>> {
    static REGISTRY: OnceLock<HashMap<String, Box<dyn Benchmark>>> = OnceLock::new();
    REGISTRY.get_or_init(|| {
        let mut map = HashMap::new();
        map.insert(
            "mmlu_pro".to_string(),
            Box::new(mmlu_pro::MmluProBenchmark) as Box<dyn Benchmark>,
        );
        map.insert(
            "kld".to_string(),
            Box::new(kld::KldBenchmark) as Box<dyn Benchmark>,
        );
        map
    })
}

pub fn get_benchmark_names() -> Vec<String> {
    registry().keys().cloned().collect()
}

pub fn pre_execute_benchmark(name: &str, config: &serde_yaml::Value) -> Result<()> {
    registry()
        .get(name)
        .ok_or_else(|| anyhow::anyhow!("Unknown benchmark: {name}"))?
        .pre_execute(config)
}

pub fn execute_benchmark(
    name: &str,
    model: &Model,
    config: &serde_yaml::Value,
) -> Result<serde_json::Value> {
    registry()
        .get(name)
        .ok_or_else(|| anyhow::anyhow!("Unknown benchmark: {name}"))?
        .execute(model, config)
}

pub fn post_execute_benchmark(
    name: &str,
    model_results: &HashMap<String, serde_json::Value>,
) -> Result<serde_json::Value> {
    registry()
        .get(name)
        .ok_or_else(|| anyhow::anyhow!("Unknown benchmark: {name}"))?
        .post_execute(model_results)
}
