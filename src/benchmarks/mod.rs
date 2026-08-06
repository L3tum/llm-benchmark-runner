use crate::config::Model;
use crate::shared::{BenchmarkCategory, BenchmarkResult, TaskResult, TestAggregate, TestName};
use anyhow::Result;
use std::collections::{BTreeMap, HashMap};
use std::sync::OnceLock;

/// Register benchmark types into the registry map. Each entry defaults the
/// benchmark type; use a plain `map.insert(...)` for types with a custom
/// constructor (e.g. `TerminalBenchBenchmark::new()`).
macro_rules! bm {
    ($map:expr, $( $name:literal => $ty:ty ),* $(,)?) => {
        $(
            $map.insert(
                $name.to_string(),
                Box::new(<$ty>::default()) as Box<dyn Benchmark>,
            );
        )*
    };
}

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
// pub mod factbench; // DISABLED: RAG-style benchmark; not useful for this suite
pub mod faithdial;
pub mod fever;
pub mod fictional_language;
pub mod gpqa;
pub mod halubench;
pub mod halueval;
pub mod harmbench;
// pub mod hdm_bench; // DISABLED: AI-generated benchmark; not useful
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
// pub mod scifact; // DISABLED: primarily for RAG benchmarks
pub mod snli;
pub mod squad_v2;
// pub mod stable_toolbench; // DISABLED: data source unavailable
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
        bm!(map, "mmlu_pro" => mmlu_pro::MmluProBenchmark);
        bm!(map, "supergpqa" => supergpqa::SuperGpqaBenchmark);
        bm!(map, "kld" => kld::KldBenchmark);
        bm!(map, "gpqa" => gpqa::GpqaBenchmark);
        bm!(map, "aime" => aime::AimeBenchmark);
        bm!(map, "apps" => apps::AppsBenchmark);
        bm!(map, "math500" => math500::Math500Benchmark);
        bm!(map, "minebench" => minebench::MinebenchBenchmark);
        bm!(map, "carwash" => carwash::CarwashBenchmark);
        bm!(map, "fictional_language" => fictional_language::FictionalLanguageBenchmark);
        bm!(map, "efficient_language" => efficient_language::EfficientLanguageBenchmark);
        bm!(map, "reverse" => reverse::ReverseBenchmark);
        bm!(map, "reverse_tools" => reverse::ReverseToolsBenchmark);
        bm!(map, "morse_code" => morse_code::MorseCodeBenchmark);
        bm!(map, "morse_code_tools" => morse_code::MorseCodeToolsBenchmark);
        bm!(map, "base64" => base64::Base64Benchmark);
        bm!(map, "base64_tools" => base64::Base64ToolsBenchmark);
        bm!(map, "hex" => hex::HexBenchmark);
        bm!(map, "hex_tools" => hex::HexToolsBenchmark);
        bm!(map, "svg_moonwalk" => svg_benchmarks::SvgMoonwalkBenchmark);
        bm!(map, "svg_bike" => svg_benchmarks::SvgBikeBenchmark);
        bm!(map, "minebench_tools" => minebench::MinebenchToolsBenchmark);
        bm!(map, "ifeval" => ifeval::IFEvalBenchmark);
        bm!(map, "harmbench" => harmbench::HarmBenchBenchmark);
        bm!(map, "coding_eval" => coding_eval::CodingEvalBenchmark);
        bm!(map, "humaneval" => coding_eval::HumanEvalBenchmark);
        bm!(map, "humaneval_plus" => coding_eval::HumanEvalPlusBenchmark);
        bm!(map, "mbpp_plus" => coding_eval::MbppPlusBenchmark);
        bm!(map, "multipl_e" => multipl_e::MultiPLEBenchmark);
        bm!(map, "swebench" => swe_bench::SweBenchBenchmark);
        bm!(map, "swebench_verified" => swe_bench::SweBenchVerifiedBenchmark);
        bm!(map, "swebench_pro" => swe_bench::SweBenchProBenchmark);
        bm!(map, "swebench_multilingual" => swe_bench::SweBenchMultilingualBenchmark);
        bm!(map, "tool_hallucination" => tool_hallucination::ToolHallucinationBenchmark);
        map.insert(
            "terminal_bench".to_string(),
            Box::new(terminal_bench::TerminalBenchBenchmark::new()) as Box<dyn Benchmark>,
        );
        bm!(map, "truthful_qa" => truthful_qa::TruthfulQABenchmark);
        bm!(map, "truthful_qa_mc2" => truthful_qa::TruthfulQAMC2Benchmark);
        bm!(map, "fever" => fever::FeverBenchmark);
        bm!(map, "halueval" => halueval::HaluEvalBenchmark);
        bm!(map, "true_false" => true_false::TrueFalseBenchmark);
        bm!(map, "faithdial" => faithdial::FaithDialBenchmark);
        bm!(map, "nq_open" => nq_open::NQOpenBenchmark);
        bm!(map, "triviaqa" => triviaqa::TriviaQABenchmark);
        bm!(map, "mmlu_pro_plus" => mmlu_pro_plus::MmluProPlusBenchmark);
        bm!(map, "mmlu_prox" => mmlu_prox::MmluProxBenchmark);
        bm!(map, "race" => race::RaceBenchmark);
        bm!(map, "squad_v2" => squad_v2::SquadV2Benchmark);
        bm!(map, "xsum" => xsum::XSumBenchmark);
        bm!(map, "cnn_dailymail" => cnn_dailymail::CnnDailyMailBenchmark);
        bm!(map, "ea_mt" => ea_mt::EAMTBenchmark);
        // --- New benchmarks: Research & Hallucination suite ---
        bm!(map, "snli" => snli::SnliBenchmark);
        bm!(map, "popqa" => popqa::PopQABenchmark);
        bm!(map, "cruxeval" => cruxeval::CruxEvalBenchmark);
        bm!(map, "ruler" => ruler::RulerBenchmark);
        bm!(map, "halubench" => halubench::HaluBenchBenchmark);
        bm!(map, "bbh" => bbh::BbhBenchmark);
        bm!(map, "bullshitbench" => bullshitbench::BullshitBenchBenchmark);
        bm!(map, "linux_kernel_security" => linux_kernel_security::LinuxKernelSecurityBenchmark);
        bm!(map, "truthful_qa_gen" => truthful_qa_gen::TruthfulQAGenBenchmark);
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
