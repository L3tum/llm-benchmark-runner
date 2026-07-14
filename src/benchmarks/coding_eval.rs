use crate::config::Model;
use crate::download::download_with_retry;
use crate::reports::model::BenchmarkResult;
use anyhow::Result;
use serde::Deserialize;
use serde_json::Value as JsonValue;
use std::collections::{BTreeMap, HashMap};
use std::fs;
use std::path::PathBuf;

pub struct CodingEvalBenchmark;
pub struct HumanEvalBenchmark;
pub struct HumanEvalPlusBenchmark;
pub struct MbppPlusBenchmark;

const HUMANEVAL_PLUS_URL: &str = "https://github.com/evalplus/humanevalplus_release/releases/download/v0.1.10/HumanEvalPlus.jsonl.gz";
const MBPP_PLUS_URL: &str =
    "https://github.com/evalplus/mbppplus_release/releases/download/v0.2.0/MbppPlus.jsonl.gz";
const HUMAN_EVAL_URL: &str =
    "https://github.com/openai/human-eval/raw/master/data/HumanEval.jsonl.gz";
const DEFAULT_DOCKER_IMAGE: &str = "python:3.12";

fn download_taskset(taskset: &TasksetConfig) -> Result<PathBuf> {
    let (url, filename) = match taskset.task_type {
        TaskType::HumanEval => (HUMAN_EVAL_URL, "HumanEval.jsonl.gz"),
        TaskType::HumanEvalPlus => (HUMANEVAL_PLUS_URL, "HumanEvalPlus.jsonl.gz"),
        TaskType::Mbpp => (MBPP_PLUS_URL, "MbppPlus.jsonl.gz"),
    };
    let cache_dir = dirs::cache_dir()
        .unwrap_or_default()
        .join("llm-benchmark-runner")
        .join("coding_eval");
    fs::create_dir_all(&cache_dir)?;
    let path = cache_dir.join(filename);
    if path.exists() {
        return Ok(path);
    }
    println!("  Downloading coding_eval taskset {}...", taskset.name);
    let bytes = download_with_retry(url, 3, 60, "llm-benchmark-runner")?
        .error_for_status()?
        .bytes()?;
    let tmp_path = path.with_extension(format!(
        "{}.tmp.{}",
        path.extension()
            .and_then(|ext| ext.to_str())
            .unwrap_or("download"),
        std::process::id()
    ));
    fs::write(&tmp_path, bytes)?;
    fs::rename(&tmp_path, &path).inspect_err(|_rename_err| {
        let _ = fs::remove_file(&tmp_path);
    })?;
    Ok(path)
}

fn parse_config(config: &yaml_serde::Value) -> Result<CodingEvalConfig> {
    let num_samples = config
        .get("num_samples")
        .and_then(|v| v.as_i64())
        .map(|v| v as usize);
    let timeout_secs = config
        .get("timeout_secs")
        .and_then(|v| v.as_i64())
        .unwrap_or(8);
    let enable_pass2 = config
        .get("enable_pass2")
        .and_then(|v| v.as_bool())
        .unwrap_or(false);
    let enable_pass3 = config
        .get("enable_pass3")
        .and_then(|v| v.as_bool())
        .unwrap_or(false);

    let mut language_images: HashMap<String, String> = HashMap::new();
    if let Some(l) = config.get("language_images").and_then(|v| v.as_mapping()) {
        for (lang, img) in l {
            if let (Some(lang_str), Some(img_str)) = (lang.as_str(), img.as_str()) {
                language_images.insert(lang_str.to_string(), img_str.to_string());
            }
        }
    }

    let tasksets: Vec<TasksetConfig> =
        if let Some(t) = config.get("tasksets").and_then(|v| v.as_sequence()) {
            t.iter()
                .filter_map(|item| {
                    let name = item.get("name")?.as_str()?.to_string();
                    let task_type_str = item.get("task_type")?.as_str()?;
                    let task_type = TaskType::from_key(task_type_str)?;
                    let language = item
                        .get("language")
                        .and_then(|v| v.as_str())
                        .unwrap_or("python")
                        .to_string();
                    let tasks_path = item
                        .get("tasks_path")
                        .and_then(|v| v.as_str())
                        .map(PathBuf::from);
                    Some(TasksetConfig {
                        name,
                        task_type,
                        language,
                        tasks_path,
                    })
                })
                .collect()
        } else {
            // Default to HumanEval if no tasksets specified
            vec![TasksetConfig {
                name: "HumanEval".to_string(),
                task_type: TaskType::HumanEval,
                language: "python".to_string(),
                tasks_path: None,
            }]
        };

    let host_repo_path = config
        .get("host_repo_path")
        .and_then(|v| v.as_str())
        .map(PathBuf::from);

    Ok(CodingEvalConfig {
        tasksets,
        num_samples,
        timeout_secs: timeout_secs as u64,
        enable_pass2,
        enable_pass3,
        language_images,
        host_repo_path,
    })
}

/// Public helper to create the common benchmark result for pass@1 coding benchmarks.
/// Used by HumanEval, HumanEval+, and MBPP+ to avoid code duplication.
pub fn common_coding_to_report_result(raw: &serde_json::Value) -> Result<BenchmarkResult> {
    // Need the trait in scope to call to_report_result
    use crate::benchmarks::Benchmark;
    CodingEvalBenchmark.to_report_result(raw)
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum TaskType {
    HumanEval,
    HumanEvalPlus,
    Mbpp,
}

impl TaskType {
    fn from_key(key: &str) -> Option<Self> {
        match key {
            "human_eval" | "humaneval" => Some(Self::HumanEval),
            "humaneval_plus" => Some(Self::HumanEvalPlus),
            "mbpp" | "mbpp_plus" => Some(Self::Mbpp),
            _ => None,
        }
    }

    fn as_str(&self) -> &'static str {
        match self {
            Self::HumanEval => "human_eval",
            Self::HumanEvalPlus => "humaneval_plus",
            Self::Mbpp => "mbpp",
        }
    }
}

#[derive(Debug, Clone)]
struct TasksetConfig {
    name: String,
    task_type: TaskType,
    language: String,
    tasks_path: Option<PathBuf>,
}

#[derive(Debug, Clone)]
struct CodingEvalConfig {
    tasksets: Vec<TasksetConfig>,
    num_samples: Option<usize>,
    timeout_secs: u64,
    enable_pass2: bool,
    enable_pass3: bool,
    language_images: HashMap<String, String>,
    host_repo_path: Option<PathBuf>,
}

#[derive(Debug, Clone, Deserialize)]
struct CodingTask {
    task_id: String,
    prompt: String,
    entry_point: String,
    #[serde(default)]
    test: Option<String>,
    #[serde(default)]
    canonical_solution: Option<String>,
    #[serde(default)]
    base_input: Option<Vec<JsonValue>>,
    #[serde(default)]
    plus_input: Option<Vec<JsonValue>>,
    #[serde(default)]
    atol: Option<f64>,
    #[serde(default)]
    #[allow(dead_code)]
    assertion: Option<String>,
}

#[derive(Debug, Clone)]
struct HarnessFiles {
    test_path: PathBuf,
}

#[derive(Debug, Clone)]
struct DockerResult {
    passed: bool,
    timed_out: bool,
    exit_code: Option<i32>,
    stdout: String,
    stderr: String,
    error_summary: String,
}

#[derive(Debug, Clone)]
struct AttemptOutcome {
    attempt: usize,
    skipped: bool,
    passed: bool,
    timed_out: bool,
    exit_code: Option<i32>,
    stdout: String,
    stderr: String,
    error_summary: String,
    output_tokens: u64,
    thinking_tokens: u64,
}

impl super::Benchmark for CodingEvalBenchmark {
    fn name(&self) -> &str {
        "coding_eval"
    }

    fn display_name(&self) -> &'static str {
        "Coding Eval"
    }

    fn category(&self) -> crate::reports::model::BenchmarkCategory {
        crate::reports::model::BenchmarkCategory::ShortContextCoding
    }

    fn pre_execute(&self, config: &yaml_serde::Value) -> Result<()> {
        let parsed = parse_config(config)?;
        for taskset in &parsed.tasksets {
            if taskset.tasks_path.is_none() {
                download_taskset(taskset)?;
            }
        }
        Ok(())
    }

    fn execute(&self, _model: &Model, _config: &yaml_serde::Value) -> Result<serde_json::Value> {
        // CodingEvalBenchmark is not meant to be run directly; use HumanEvalBenchmark etc.
        Err(anyhow::anyhow!("CodingEvalBenchmark execute: not directly runnable. Use HumanEval, HumanEval+, or MBPP+ instead."))
    }

    fn to_report_result(&self, raw: &serde_json::Value) -> Result<BenchmarkResult> {
        use crate::reports::model::{BreakdownTable, Score, ScoreUnit};

        let pass_at_1 = raw.get("pass_at_1").and_then(|v| v.as_f64()).unwrap_or(0.0);
        let pass_at_2 = raw.get("pass_at_2").and_then(|v| v.as_f64());
        let pass_at_3 = raw.get("pass_at_3").and_then(|v| v.as_f64());
        let passed = raw.get("passed").and_then(|v| v.as_i64()).unwrap_or(0);
        let total_questions = raw
            .get("total_questions")
            .and_then(|v| v.as_i64())
            .unwrap_or(0);
        let timeout_count = raw
            .get("timeout_count")
            .and_then(|v| v.as_i64())
            .unwrap_or(0);
        let output_tokens = raw
            .get("output_tokens")
            .and_then(|v| v.as_i64())
            .unwrap_or(0);
        let thinking_tokens = raw
            .get("thinking_tokens")
            .and_then(|v| v.as_i64())
            .unwrap_or(0);

        // Build common scores
        let mut scores = BTreeMap::new();
        scores.insert(
            "pass_at_1".to_string(),
            Score::float(pass_at_1, ScoreUnit::Percent)
                .primary(true)
                .higher_is_better(true),
        );
        if let Some(pass2) = pass_at_2 {
            scores.insert(
                "pass_at_2".to_string(),
                Score::float(pass2, ScoreUnit::Percent),
            );
        }
        if let Some(pass3) = pass_at_3 {
            scores.insert(
                "pass_at_3".to_string(),
                Score::float(pass3, ScoreUnit::Percent),
            );
        }
        scores.insert(
            "passed".to_string(),
            Score::integer(passed, ScoreUnit::Count),
        );
        scores.insert(
            "total_questions".to_string(),
            Score::integer(total_questions, ScoreUnit::Count),
        );
        if output_tokens > 0 {
            scores.insert(
                "output_tokens".to_string(),
                Score::integer(output_tokens, ScoreUnit::Tokens),
            );
        }
        if thinking_tokens > 0 {
            scores.insert(
                "thinking_tokens".to_string(),
                Score::integer(thinking_tokens, ScoreUnit::Tokens),
            );
        }

        // Add breakdown table for pass@k scores (row = model pass rate, columns = metrics)
        let mut breakdown_table_rows = BTreeMap::new();
        let mut row_scores = BTreeMap::new();
        row_scores.insert(
            "pass@1".to_string(),
            Score::float(pass_at_1, ScoreUnit::Percent),
        );
        if let Some(pass2) = pass_at_2 {
            row_scores.insert(
                "pass@2".to_string(),
                Score::float(pass2, ScoreUnit::Percent),
            );
        }
        if let Some(pass3) = pass_at_3 {
            row_scores.insert(
                "pass@3".to_string(),
                Score::float(pass3, ScoreUnit::Percent),
            );
        }
        row_scores.insert(
            "timeout_count".to_string(),
            Score::integer(timeout_count, ScoreUnit::Count),
        );
        breakdown_table_rows.insert("pass_k".to_string(), row_scores);

        let breakdowns = if !breakdown_table_rows.is_empty() {
            BTreeMap::from([(
                "pass_k".to_string(),
                BreakdownTable {
                    title: "Pass@k Breakdown".to_string(),
                    rows: breakdown_table_rows,
                },
            )])
        } else {
            BTreeMap::new()
        };

        Ok(BenchmarkResult {
            scores,
            breakdowns,
            artifacts: vec![],
            diagnostics: vec![],
            raw: raw.clone(),
        })
    }
}

/// Public helper to create the common benchmark result for pass@1 coding benchmarks.
/// Used by HumanEval, HumanEval+, and MBPP+ to avoid code duplication.
/// Create a config with a single preset taskset while preserving other config values.
fn preset_config(config: &yaml_serde::Value, name: &str, task_type: TaskType) -> yaml_serde::Value {
    let mut map = if let Some(m) = config.as_mapping() {
        m.clone()
    } else {
        yaml_serde::Mapping::new()
    };
    let mut taskset = yaml_serde::Mapping::new();
    taskset.insert(
        yaml_serde::Value::String("name".into()),
        yaml_serde::Value::String(name.into()),
    );
    taskset.insert(
        yaml_serde::Value::String("task_type".into()),
        yaml_serde::Value::String(task_type.as_str().into()),
    );
    let tasksets = yaml_serde::Value::Sequence(vec![yaml_serde::Value::Mapping(taskset)]);
    map.insert(yaml_serde::Value::String("tasksets".into()), tasksets);
    yaml_serde::Value::Mapping(map)
}

impl super::Benchmark for HumanEvalBenchmark {
    fn name(&self) -> &str {
        "humaneval"
    }

    fn display_name(&self) -> &'static str {
        "HumanEval"
    }

    fn category(&self) -> crate::reports::model::BenchmarkCategory {
        crate::reports::model::BenchmarkCategory::ShortContextCoding
    }

    fn to_report_result(&self, raw: &serde_json::Value) -> Result<BenchmarkResult> {
        // Delegate to the common implementation in CodingEvalBenchmark
        common_coding_to_report_result(raw)
    }

    fn pre_execute(&self, config: &yaml_serde::Value) -> Result<()> {
        let cfg = preset_config(config, "humaneval", TaskType::HumanEval);
        CodingEvalBenchmark.pre_execute(&cfg)
    }

    fn execute(&self, model: &Model, config: &yaml_serde::Value) -> Result<serde_json::Value> {
        let cfg = preset_config(config, "humaneval", TaskType::HumanEval);
        CodingEvalBenchmark.execute(model, &cfg)
    }
}

impl super::Benchmark for HumanEvalPlusBenchmark {
    fn name(&self) -> &str {
        "humaneval_plus"
    }

    fn display_name(&self) -> &'static str {
        "HumanEval+"
    }

    fn category(&self) -> crate::reports::model::BenchmarkCategory {
        crate::reports::model::BenchmarkCategory::ShortContextCoding
    }

    fn to_report_result(&self, raw: &serde_json::Value) -> Result<BenchmarkResult> {
        // Delegate to the common implementation in CodingEvalBenchmark
        common_coding_to_report_result(raw)
    }

    fn pre_execute(&self, config: &yaml_serde::Value) -> Result<()> {
        let cfg = preset_config(config, "humaneval_plus", TaskType::HumanEvalPlus);
        CodingEvalBenchmark.pre_execute(&cfg)
    }

    fn execute(&self, model: &Model, config: &yaml_serde::Value) -> Result<serde_json::Value> {
        let cfg = preset_config(config, "humaneval_plus", TaskType::HumanEvalPlus);
        CodingEvalBenchmark.execute(model, &cfg)
    }
}

impl super::Benchmark for MbppPlusBenchmark {
    fn name(&self) -> &str {
        "mbpp_plus"
    }

    fn display_name(&self) -> &'static str {
        "MBPP+"
    }

    fn category(&self) -> crate::reports::model::BenchmarkCategory {
        crate::reports::model::BenchmarkCategory::ShortContextCoding
    }

    fn to_report_result(&self, raw: &serde_json::Value) -> Result<BenchmarkResult> {
        // Delegate to the common implementation in CodingEvalBenchmark
        common_coding_to_report_result(raw)
    }

    fn pre_execute(&self, config: &yaml_serde::Value) -> Result<()> {
        let cfg = preset_config(config, "mbpp_plus", TaskType::Mbpp);
        CodingEvalBenchmark.pre_execute(&cfg)
    }

    fn execute(&self, model: &Model, config: &yaml_serde::Value) -> Result<serde_json::Value> {
        let cfg = preset_config(config, "mbpp_plus", TaskType::Mbpp);
        CodingEvalBenchmark.execute(model, &cfg)
    }
}
