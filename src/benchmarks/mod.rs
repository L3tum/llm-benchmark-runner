use crate::config::Model;
use crate::reports::model::{BenchmarkCategory, BenchmarkResult, TestAggregate, TestName};
use anyhow::Result;
use std::collections::HashMap;
use std::sync::OnceLock;

pub mod aime;
pub mod answer_classifier;
pub mod carwash;
pub mod cnn_dailymail;
pub mod coding_eval;
pub mod ea_mt;
pub mod faithdial;
pub mod fever;
pub mod gpqa;
pub mod halueval;
pub mod harmbench;
pub mod hdm_bench;
pub mod ifeval;
pub mod kld;
pub mod math500;
pub mod minebench;
pub mod mmlu_pro;
pub mod mmlu_pro_plus;
pub mod mmlu_prox;
pub mod nq_open;
pub mod race;
pub mod squad_v2;
pub mod supergpqa;
pub mod swe_bench;
pub mod tool_hallucination;
pub mod triviaqa;
pub mod true_false;
pub mod truthful_qa;
pub mod xsum;

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

    fn pre_execute(&self, _config: &yaml_serde::Value) -> Result<()> {
        Ok(())
    }
    /// Execute the benchmark for a single model, returning a normalized result with scores.
    /// The `raw` field of the returned `BenchmarkResult` should contain the serialized
    /// JSON that would have been returned previously, for backwards compatibility and debug.
    fn execute(&self, model: &Model, config: &yaml_serde::Value) -> Result<BenchmarkResult>;

    /// Convert an in-memory `BenchmarkResult` (from a previous run or resume) into a
    /// `BenchmarkResult` for reports. Should typically just return `Ok(b.clone())`.
    fn to_report_result(&self, b: &BenchmarkResult) -> Result<BenchmarkResult> {
        Ok(b.clone())
    }

    /// For benchmarks that produce aggregates (e.g., KLD pairwise) from all model results,
    /// convert the result of `post_execute` into a `TestAggregate`. Default: None.
    fn to_report_aggregate(&self, _result: &BenchmarkResult) -> Result<Option<TestAggregate>> {
        Ok(None)
    }

    /// Post-execute processing that combines all models' results for this benchmark.
    /// Returns a `BenchmarkResult` containing aggregate metrics (e.g., KLD pairwise).
    fn post_execute(
        &self,
        _model_results: &HashMap<String, BenchmarkResult>,
    ) -> Result<BenchmarkResult> {
        Ok(BenchmarkResult::empty())
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
            "supergpqa".to_string(),
            Box::new(supergpqa::SuperGpqaBenchmark) as Box<dyn Benchmark>,
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
        map.insert(
            "tool_hallucination".to_string(),
            Box::new(tool_hallucination::ToolHallucinationBenchmark) as Box<dyn Benchmark>,
        );
        map.insert(
            "truthful_qa".to_string(),
            Box::new(truthful_qa::TruthfulQABenchmark) as Box<dyn Benchmark>,
        );
        map.insert(
            "truthful_qa_mc2".to_string(),
            Box::new(truthful_qa::TruthfulQAMC2Benchmark) as Box<dyn Benchmark>,
        );
        map.insert(
            "fever".to_string(),
            Box::new(fever::FeverBenchmark) as Box<dyn Benchmark>,
        );
        map.insert(
            "halueval".to_string(),
            Box::new(halueval::HaluEvalBenchmark) as Box<dyn Benchmark>,
        );
        map.insert(
            "true_false".to_string(),
            Box::new(true_false::TrueFalseBenchmark) as Box<dyn Benchmark>,
        );
        map.insert(
            "faithdial".to_string(),
            Box::new(faithdial::FaithDialBenchmark) as Box<dyn Benchmark>,
        );
        map.insert(
            "hdm_bench".to_string(),
            Box::new(hdm_bench::HdmBenchBenchmark) as Box<dyn Benchmark>,
        );
        map.insert(
            "nq_open".to_string(),
            Box::new(nq_open::NQOpenBenchmark) as Box<dyn Benchmark>,
        );
        map.insert(
            "triviaqa".to_string(),
            Box::new(triviaqa::TriviaQABenchmark) as Box<dyn Benchmark>,
        );
        map.insert(
            "mmlu_pro_plus".to_string(),
            Box::new(mmlu_pro_plus::MmluProPlusBenchmark) as Box<dyn Benchmark>,
        );
        map.insert(
            "mmlu_prox".to_string(),
            Box::new(mmlu_prox::MmluProxBenchmark) as Box<dyn Benchmark>,
        );
        map.insert(
            "race".to_string(),
            Box::new(race::RaceBenchmark) as Box<dyn Benchmark>,
        );
        map.insert(
            "squad_v2".to_string(),
            Box::new(squad_v2::SquadV2Benchmark) as Box<dyn Benchmark>,
        );
        map.insert(
            "xsum".to_string(),
            Box::new(xsum::XSumBenchmark) as Box<dyn Benchmark>,
        );
        map.insert(
            "cnn_dailymail".to_string(),
            Box::new(cnn_dailymail::CnnDailyMailBenchmark) as Box<dyn Benchmark>,
        );
        map.insert(
            "ea_mt".to_string(),
            Box::new(ea_mt::EAMTBenchmark) as Box<dyn Benchmark>,
        );
        map
    })
}

pub fn get_benchmark_names() -> Vec<String> {
    registry().keys().cloned().collect()
}

pub fn pre_execute_benchmark(name: &str, config: &yaml_serde::Value) -> Result<()> {
    registry()
        .get(name)
        .ok_or_else(|| anyhow::anyhow!("Unknown benchmark: {name}"))?
        .pre_execute(config)
}

pub fn execute_benchmark(
    name: &str,
    model: &Model,
    config: &yaml_serde::Value,
) -> Result<BenchmarkResult> {
    registry()
        .get(name)
        .ok_or_else(|| anyhow::anyhow!("Unknown benchmark: {name}"))?
        .execute(model, config)
}

pub fn post_execute_benchmark(
    name: &str,
    model_results: &HashMap<String, BenchmarkResult>,
) -> Result<BenchmarkResult> {
    registry()
        .get(name)
        .ok_or_else(|| anyhow::anyhow!("Unknown benchmark: {name}"))?
        .post_execute(model_results)
}
