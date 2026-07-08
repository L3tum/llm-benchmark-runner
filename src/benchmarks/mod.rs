use crate::config::Model;
use crate::reports::model::{BenchmarkCategory, BenchmarkResult, TestAggregate, TestName};
use anyhow::Result;
use std::collections::HashMap;
use std::sync::OnceLock;

pub mod aime;
pub mod carwash;
pub mod coding_eval;
pub mod gpqa;
pub mod harmbench;
pub mod ifeval;
pub mod kld;
pub mod math500;
pub mod minebench;
pub mod mmlu_pro;
pub mod swe_bench;

/// Trait for all benchmarks.
pub trait Benchmark: Send + Sync {
    fn name(&self) -> &str;
    /// Normalized test name for report data.
    fn test_name(&self) -> TestName {
        TestName::new(self.name().to_string())
    }
    /// Human-readable display name.
    fn display_name(&self) -> &'static str;
    /// Category for report grouping.
    fn category(&self) -> BenchmarkCategory;

    fn pre_execute(&self, _config: &serde_yaml::Value) -> Result<()> {
        Ok(())
    }
    fn execute(&self, model: &Model, config: &serde_yaml::Value) -> Result<serde_json::Value>;
    /// Convert the raw JSON result into a normalized `BenchmarkResult`.
    /// Default implementation returns `Ok(BenchmarkResult::empty())` — benchmarks should override.
    fn to_report_result(&self, _raw: &serde_json::Value) -> Result<BenchmarkResult> {
        Ok(BenchmarkResult::empty())
    }
    /// Convert the raw post-execute JSON into an optional aggregate.
    /// Default returns `Ok(None)` — benchmarks like KLD should override.
    fn to_report_aggregate(&self, _raw: &serde_json::Value) -> Result<Option<TestAggregate>> {
        Ok(None)
    }
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
        map.insert(
            "gpqa".to_string(),
            Box::new(gpqa::GpqaBenchmark) as Box<dyn Benchmark>,
        );
        map.insert(
            "aime".to_string(),
            Box::new(aime::AimeBenchmark) as Box<dyn Benchmark>,
        );
        map.insert(
            "math500".to_string(),
            Box::new(math500::Math500Benchmark) as Box<dyn Benchmark>,
        );
        map.insert(
            "minebench".to_string(),
            Box::new(minebench::MinebenchBenchmark) as Box<dyn Benchmark>,
        );
        map.insert(
            "carwash".to_string(),
            Box::new(carwash::CarwashBenchmark) as Box<dyn Benchmark>,
        );
        map.insert(
            "ifeval".to_string(),
            Box::new(ifeval::IFEvalBenchmark) as Box<dyn Benchmark>,
        );
        map.insert(
            "harmbench".to_string(),
            Box::new(harmbench::HarmBenchBenchmark) as Box<dyn Benchmark>,
        );
        map.insert(
            "coding_eval".to_string(),
            Box::new(coding_eval::CodingEvalBenchmark) as Box<dyn Benchmark>,
        );
        map.insert(
            "humaneval".to_string(),
            Box::new(coding_eval::HumanEvalBenchmark) as Box<dyn Benchmark>,
        );
        map.insert(
            "humaneval_plus".to_string(),
            Box::new(coding_eval::HumanEvalPlusBenchmark) as Box<dyn Benchmark>,
        );
        map.insert(
            "mbpp_plus".to_string(),
            Box::new(coding_eval::MbppPlusBenchmark) as Box<dyn Benchmark>,
        );
        map.insert(
            "swebench".to_string(),
            Box::new(swe_bench::SweBenchBenchmark) as Box<dyn Benchmark>,
        );
        map.insert(
            "swebench_verified".to_string(),
            Box::new(swe_bench::SweBenchVerifiedBenchmark) as Box<dyn Benchmark>,
        );
        map.insert(
            "swebench_pro".to_string(),
            Box::new(swe_bench::SweBenchProBenchmark) as Box<dyn Benchmark>,
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
