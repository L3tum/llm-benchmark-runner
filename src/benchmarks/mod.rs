use crate::config::Model;
use crate::reports::model::{
    BenchmarkCategory, BenchmarkResult, TaskResult, TestAggregate, TestName,
};
use anyhow::Result;
use std::collections::HashMap;
use std::sync::OnceLock;

pub mod aime;
pub mod answer_classifier;
pub mod base64;
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
pub mod hex;
pub mod ifeval;
pub mod kld;
pub mod math500;
pub mod minebench;
pub mod mmlu_pro;
pub mod mmlu_pro_plus;
pub mod mmlu_prox;
pub mod morse_code;
pub mod nq_open;
pub mod race;
pub mod reverse;
pub mod squad_v2;
pub mod stable_toolbench;
pub mod supergpqa;
pub mod svg_benchmarks;
pub mod swe_bench;
pub mod terminal_bench;
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
    /// Execute a single task, returning `Ok(Some(TaskResult))` or `Ok(None)` when done.
    /// The benchmark manages its own iteration state internally (uses Mutex for interior mutability).
    /// The `tracker` is owned by the runner; the benchmark should use it for all LLM calls.
    /// Token counts are accumulated by the runner from tracker snapshots — the returned
    /// `TaskResult` should leave `output_tokens` and `thinking_tokens` at their default `0`.
    fn execute_one(
        &self,
        model: &Model,
        config: &yaml_serde::Value,
        tracker: &mut crate::token_tracker::TokenTracker,
    ) -> Result<Option<TaskResult>>;

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
            Box::new(mmlu_pro::MmluProBenchmark::default()) as Box<dyn Benchmark>,
        );
        map.insert(
            "supergpqa".to_string(),
            Box::new(supergpqa::SuperGpqaBenchmark::default()) as Box<dyn Benchmark>,
        );
        map.insert(
            "kld".to_string(),
            Box::new(kld::KldBenchmark::default()) as Box<dyn Benchmark>,
        );
        map.insert(
            "gpqa".to_string(),
            Box::new(gpqa::GpqaBenchmark::default()) as Box<dyn Benchmark>,
        );
        map.insert(
            "aime".to_string(),
            Box::new(aime::AimeBenchmark::default()) as Box<dyn Benchmark>,
        );
        map.insert(
            "math500".to_string(),
            Box::new(math500::Math500Benchmark::default()) as Box<dyn Benchmark>,
        );
        map.insert(
            "minebench".to_string(),
            Box::new(minebench::MinebenchBenchmark::default()) as Box<dyn Benchmark>,
        );
        map.insert(
            "carwash".to_string(),
            Box::new(carwash::CarwashBenchmark::default()) as Box<dyn Benchmark>,
        );
        map.insert(
            "reverse".to_string(),
            Box::new(reverse::ReverseBenchmark::default()) as Box<dyn Benchmark>,
        );
        map.insert(
            "reverse_tools".to_string(),
            Box::new(reverse::ReverseToolsBenchmark::default()) as Box<dyn Benchmark>,
        );
        map.insert(
            "morse_code".to_string(),
            Box::new(morse_code::MorseCodeBenchmark::default()) as Box<dyn Benchmark>,
        );
        map.insert(
            "morse_code_tools".to_string(),
            Box::new(morse_code::MorseCodeToolsBenchmark::default()) as Box<dyn Benchmark>,
        );
        map.insert(
            "base64".to_string(),
            Box::new(base64::Base64Benchmark::default()) as Box<dyn Benchmark>,
        );
        map.insert(
            "base64_tools".to_string(),
            Box::new(base64::Base64ToolsBenchmark::default()) as Box<dyn Benchmark>,
        );
        map.insert(
            "hex".to_string(),
            Box::new(hex::HexBenchmark::default()) as Box<dyn Benchmark>,
        );
        map.insert(
            "hex_tools".to_string(),
            Box::new(hex::HexToolsBenchmark::default()) as Box<dyn Benchmark>,
        );
        map.insert(
            "svg_moonwalk".to_string(),
            Box::new(svg_benchmarks::SvgMoonwalkBenchmark::default()) as Box<dyn Benchmark>,
        );
        map.insert(
            "svg_bike".to_string(),
            Box::new(svg_benchmarks::SvgBikeBenchmark::default()) as Box<dyn Benchmark>,
        );
        map.insert(
            "minebench_tools".to_string(),
            Box::new(minebench::MinebenchToolsBenchmark::default()) as Box<dyn Benchmark>,
        );
        map.insert(
            "ifeval".to_string(),
            Box::new(ifeval::IFEvalBenchmark::default()) as Box<dyn Benchmark>,
        );
        map.insert(
            "harmbench".to_string(),
            Box::new(harmbench::HarmBenchBenchmark::default()) as Box<dyn Benchmark>,
        );
        map.insert(
            "coding_eval".to_string(),
            Box::new(coding_eval::CodingEvalBenchmark::default()) as Box<dyn Benchmark>,
        );
        map.insert(
            "humaneval".to_string(),
            Box::new(coding_eval::HumanEvalBenchmark::default()) as Box<dyn Benchmark>,
        );
        map.insert(
            "humaneval_plus".to_string(),
            Box::new(coding_eval::HumanEvalPlusBenchmark::default()) as Box<dyn Benchmark>,
        );
        map.insert(
            "mbpp_plus".to_string(),
            Box::new(coding_eval::MbppPlusBenchmark::default()) as Box<dyn Benchmark>,
        );
        map.insert(
            "swebench".to_string(),
            Box::new(swe_bench::SweBenchBenchmark::default()) as Box<dyn Benchmark>,
        );
        map.insert(
            "swebench_verified".to_string(),
            Box::new(swe_bench::SweBenchVerifiedBenchmark::default()) as Box<dyn Benchmark>,
        );
        map.insert(
            "swebench_pro".to_string(),
            Box::new(swe_bench::SweBenchProBenchmark::default()) as Box<dyn Benchmark>,
        );
        map.insert(
            "tool_hallucination".to_string(),
            Box::new(tool_hallucination::ToolHallucinationBenchmark::default())
                as Box<dyn Benchmark>,
        );
        map.insert(
            "terminal_bench".to_string(),
            Box::new(terminal_bench::TerminalBenchBenchmark::new()) as Box<dyn Benchmark>,
        );
        map.insert(
            "stable_toolbench".to_string(),
            Box::new(stable_toolbench::StableToolBenchBenchmark::new()) as Box<dyn Benchmark>,
        );
        map.insert(
            "truthful_qa".to_string(),
            Box::new(truthful_qa::TruthfulQABenchmark::default()) as Box<dyn Benchmark>,
        );
        map.insert(
            "truthful_qa_mc2".to_string(),
            Box::new(truthful_qa::TruthfulQAMC2Benchmark::default()) as Box<dyn Benchmark>,
        );
        map.insert(
            "fever".to_string(),
            Box::new(fever::FeverBenchmark::default()) as Box<dyn Benchmark>,
        );
        map.insert(
            "halueval".to_string(),
            Box::new(halueval::HaluEvalBenchmark::default()) as Box<dyn Benchmark>,
        );
        map.insert(
            "true_false".to_string(),
            Box::new(true_false::TrueFalseBenchmark::default()) as Box<dyn Benchmark>,
        );
        map.insert(
            "faithdial".to_string(),
            Box::new(faithdial::FaithDialBenchmark::default()) as Box<dyn Benchmark>,
        );
        map.insert(
            "hdm_bench".to_string(),
            Box::new(hdm_bench::HdmBenchBenchmark::default()) as Box<dyn Benchmark>,
        );
        map.insert(
            "nq_open".to_string(),
            Box::new(nq_open::NQOpenBenchmark::default()) as Box<dyn Benchmark>,
        );
        map.insert(
            "triviaqa".to_string(),
            Box::new(triviaqa::TriviaQABenchmark::default()) as Box<dyn Benchmark>,
        );
        map.insert(
            "mmlu_pro_plus".to_string(),
            Box::new(mmlu_pro_plus::MmluProPlusBenchmark::default()) as Box<dyn Benchmark>,
        );
        map.insert(
            "mmlu_prox".to_string(),
            Box::new(mmlu_prox::MmluProxBenchmark::default()) as Box<dyn Benchmark>,
        );
        map.insert(
            "race".to_string(),
            Box::new(race::RaceBenchmark::default()) as Box<dyn Benchmark>,
        );
        map.insert(
            "squad_v2".to_string(),
            Box::new(squad_v2::SquadV2Benchmark::default()) as Box<dyn Benchmark>,
        );
        map.insert(
            "xsum".to_string(),
            Box::new(xsum::XSumBenchmark::default()) as Box<dyn Benchmark>,
        );
        map.insert(
            "cnn_dailymail".to_string(),
            Box::new(cnn_dailymail::CnnDailyMailBenchmark::default()) as Box<dyn Benchmark>,
        );
        map.insert(
            "ea_mt".to_string(),
            Box::new(ea_mt::EAMTBenchmark::default()) as Box<dyn Benchmark>,
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

/// Execute one task from a benchmark. Returns `Ok(Some(TaskResult))` or `Ok(None)` when done.
pub fn execute_benchmark_one(
    name: &str,
    model: &Model,
    config: &yaml_serde::Value,
    tracker: &mut crate::token_tracker::TokenTracker,
) -> Result<Option<TaskResult>> {
    registry()
        .get(name)
        .ok_or_else(|| anyhow::anyhow!("Unknown benchmark: {name}"))?
        .execute_one(model, config, tracker)
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
