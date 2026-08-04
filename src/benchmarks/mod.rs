use crate::config::Model;
use crate::shared::{BenchmarkCategory, BenchmarkResult, TaskResult, TestAggregate, TestName};
use anyhow::Result;
use std::collections::{BTreeMap, HashMap};
use std::sync::OnceLock;

// Re-export shared translation benchmark types from shared module.
pub use crate::shared::{Difficulty, TranslationState};

pub mod aime;
pub mod answer_classifier;
pub mod apps;
pub mod base64;
pub mod bbh;
pub mod bullshitbench;
pub mod carwash;
pub mod cnn_dailymail;
pub mod coding_eval;
pub mod cruxeval;
pub mod ea_mt;
pub mod efficient_language;
pub mod factbench;
pub mod faithdial;
pub mod fever;
pub mod fictional_language;
pub mod gpqa;
pub mod halubench;
pub mod halueval;
pub mod harmbench;
pub mod hdm_bench;
pub mod hex;
pub mod ifeval;
pub mod kld;
pub mod linux_kernel_security;
pub mod math500;
pub mod minebench;
pub mod mmlu_pro;
pub mod mmlu_pro_plus;
pub mod mmlu_prox;
pub mod morse_code;
pub mod multipl_e;
pub mod nq_open;
pub mod popqa;
pub mod race;
pub mod reverse;
pub mod ruler;
pub mod scifact;
pub mod snli;
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
pub mod truthful_qa_gen;
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

    /// Called once after all `execute_one` calls complete.
    /// Override when your benchmark needs batch evaluation
    /// (e.g., Docker evaluation, patch application + test running).
    ///
    /// `task_results` contains N results from the `execute_one` loop,
    /// each with `output_tokens`/`thinking_tokens` already populated by the runner.
    ///
    /// Return `Some(new_results)` to replace the task results.
    /// Return `Ok(None)` to keep the original results unchanged.
    ///
    /// When returning new results, preserve `output_tokens`/`thinking_tokens`
    /// from the corresponding original result by matching `task_id`.
    ///
    /// Default: does nothing (returns `Ok(None)`).
    fn batch_evaluate(
        &self,
        _task_results: &[TaskResult],
        _config: &yaml_serde::Value,
    ) -> Result<Option<Vec<TaskResult>>> {
        Ok(None)
    }
}

fn registry() -> &'static BTreeMap<String, Box<dyn Benchmark>> {
    static REGISTRY: OnceLock<BTreeMap<String, Box<dyn Benchmark>>> = OnceLock::new();
    REGISTRY.get_or_init(|| {
        let mut map = BTreeMap::new();
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
            "apps".to_string(),
            Box::new(apps::AppsBenchmark::default()) as Box<dyn Benchmark>,
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
            "fictional_language".to_string(),
            Box::new(fictional_language::FictionalLanguageBenchmark::default())
                as Box<dyn Benchmark>,
        );
        map.insert(
            "efficient_language".to_string(),
            Box::new(efficient_language::EfficientLanguageBenchmark::default())
                as Box<dyn Benchmark>,
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
            "multipl_e".to_string(),
            Box::new(multipl_e::MultiPLEBenchmark::default()) as Box<dyn Benchmark>,
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
            "swebench_multilingual".to_string(),
            Box::new(swe_bench::SweBenchMultilingualBenchmark::default()) as Box<dyn Benchmark>,
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
        // --- New benchmarks: Research & Hallucination suite ---
        map.insert(
            "scifact".to_string(),
            Box::new(scifact::SciFactBenchmark::default()) as Box<dyn Benchmark>,
        );
        map.insert(
            "snli".to_string(),
            Box::new(snli::SnliBenchmark::default()) as Box<dyn Benchmark>,
        );
        map.insert(
            "popqa".to_string(),
            Box::new(popqa::PopQABenchmark::default()) as Box<dyn Benchmark>,
        );
        map.insert(
            "cruxeval".to_string(),
            Box::new(cruxeval::CruxEvalBenchmark::default()) as Box<dyn Benchmark>,
        );
        map.insert(
            "ruler".to_string(),
            Box::new(ruler::RulerBenchmark::default()) as Box<dyn Benchmark>,
        );
        map.insert(
            "halubench".to_string(),
            Box::new(halubench::HaluBenchBenchmark::default()) as Box<dyn Benchmark>,
        );
        map.insert(
            "bbh".to_string(),
            Box::new(bbh::BbhBenchmark::default()) as Box<dyn Benchmark>,
        );
        map.insert(
            "bullshitbench".to_string(),
            Box::new(bullshitbench::BullshitBenchBenchmark::default()) as Box<dyn Benchmark>,
        );
        map.insert(
            "linux_kernel_security".to_string(),
            Box::new(linux_kernel_security::LinuxKernelSecurityBenchmark::default())
                as Box<dyn Benchmark>,
        );
        map.insert(
            "factbench".to_string(),
            Box::new(factbench::FactBenchBenchmark::default()) as Box<dyn Benchmark>,
        );
        map.insert(
            "truthful_qa_gen".to_string(),
            Box::new(truthful_qa_gen::TruthfulQAGenBenchmark::default()) as Box<dyn Benchmark>,
        );
        map
    })
}

pub fn get_benchmark_names() -> Vec<String> {
    registry().keys().cloned().collect()
}

/// Get a reference to a benchmark by name (for trait dispatch).
pub fn get_benchmark(name: &str) -> Result<&'static dyn Benchmark> {
    registry()
        .get(name)
        .ok_or_else(|| anyhow::anyhow!("Benchmark '{}' not found", name))
        .map(|b| b.as_ref() as &dyn Benchmark)
}

/// Iterate over all registered benchmarks, yielding (name, &Box<dyn Benchmark>).
/// Order is alphabetical (backed by BTreeMap).
pub fn iter_benchmarks() -> impl Iterator<Item = (&'static String, &'static Box<dyn Benchmark>)> {
    registry().iter()
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn iter_benchmarks_returns_all_registered() {
        let count = iter_benchmarks().count();
        assert_eq!(count, get_benchmark_names().len());
        // Verify we can access benchmarks by name
        for (name, _bench) in iter_benchmarks() {
            assert!(!name.is_empty(), "Benchmark name should not be empty");
        }
    }

    // ---- batch_evaluate path tests ----

    /// Mock benchmark for testing batch_evaluate trait method behavior.
    struct MockBatchBenchmark {
        mode: MockBatchMode,
    }

    enum MockBatchMode {
        NewResults,
        NoOp,
        Error,
    }

    impl Benchmark for MockBatchBenchmark {
        fn name(&self) -> &str {
            "mock_batch"
        }
        fn display_name(&self) -> &'static str {
            "Mock Batch"
        }
        fn category(&self) -> BenchmarkCategory {
            BenchmarkCategory::Other("test".into())
        }
        fn execute_one(
            &self,
            _model: &Model,
            _config: &yaml_serde::Value,
            _tracker: &mut crate::token_tracker::TokenTracker,
        ) -> Result<Option<TaskResult>> {
            Ok(None)
        }

        fn batch_evaluate(
            &self,
            task_results: &[TaskResult],
            _config: &yaml_serde::Value,
        ) -> Result<Option<Vec<TaskResult>>> {
            match self.mode {
                MockBatchMode::NewResults => {
                    // Replace results — mark all as passed
                    let new_results: Vec<TaskResult> = task_results
                        .iter()
                        .map(|t| {
                            let mut t = t.clone();
                            t.passed = true;
                            t.score = 1.0;
                            t
                        })
                        .collect();
                    Ok(Some(new_results))
                }
                MockBatchMode::NoOp => Ok(None),
                MockBatchMode::Error => Err(anyhow::anyhow!("batch eval failed for testing")),
            }
        }
    }

    #[test]
    fn batch_evaluate_can_replace_results() {
        let bench = MockBatchBenchmark {
            mode: MockBatchMode::NewResults,
        };
        let task_results = vec![
            TaskResult::new("t1", false, 0.0, vec![]),
            TaskResult::new("t2", false, 0.0, vec![]),
        ];

        let result = bench
            .batch_evaluate(&task_results, &yaml_serde::Value::Null)
            .unwrap();
        assert!(result.is_some());
        let updated = result.unwrap();
        assert_eq!(updated.len(), 2);
        assert!(updated[0].passed);
        assert!(updated[1].passed);
    }

    #[test]
    fn batch_evaluate_can_return_none_noop() {
        let bench = MockBatchBenchmark {
            mode: MockBatchMode::NoOp,
        };
        let task_results = vec![TaskResult::new("t1", true, 1.0, vec![])];

        let result = bench
            .batch_evaluate(&task_results, &yaml_serde::Value::Null)
            .unwrap();
        assert!(result.is_none());
    }

    #[test]
    fn batch_evaluate_can_return_error() {
        let bench = MockBatchBenchmark {
            mode: MockBatchMode::Error,
        };
        let task_results = vec![TaskResult::new("t1", true, 1.0, vec![])];

        let result = bench.batch_evaluate(&task_results, &yaml_serde::Value::Null);
        assert!(result.is_err());
    }
}
