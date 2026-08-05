use crate::benchmarks::Benchmark;
use crate::config::Model;
use crate::docker_runner::{DockerMount, DockerRunConfig, DockerRunner};
use crate::download::download_with_retry_bytes_sha256;
use crate::shared::{BenchmarkCategory, BenchmarkResult, Score, ScoreUnit, TaskResult};
use crate::token_tracker::TokenTracker;
use anyhow::Result;
use serde::Deserialize;
use serde_json::Value as JsonValue;
use std::collections::{BTreeMap, HashMap};
use std::fs;
use std::io::Write;
use std::path::PathBuf;
use std::sync::Mutex;

pub struct CodingEvalBenchmark {
    state: Mutex<CodingEvalState>,
}

struct CodingEvalState {
    items: Vec<serde_json::Value>,
    current_idx: usize,
    config: Option<CodingEvalConfig>,
}

impl Default for CodingEvalBenchmark {
    fn default() -> Self {
        Self {
            state: Mutex::new(CodingEvalState {
                items: Vec::new(),
                current_idx: 0,
                config: None,
            }),
        }
    }
}
pub struct HumanEvalBenchmark {
    state: Mutex<HumanEvalState>,
}

struct HumanEvalState {
    items: Vec<serde_json::Value>,
    current_idx: usize,
    config: Option<CodingEvalConfig>,
}

impl Default for HumanEvalBenchmark {
    fn default() -> Self {
        Self {
            state: Mutex::new(HumanEvalState {
                items: Vec::new(),
                current_idx: 0,
                config: None,
            }),
        }
    }
}
pub struct HumanEvalPlusBenchmark {
    state: Mutex<HumanEvalPlusState>,
}

struct HumanEvalPlusState {
    items: Vec<serde_json::Value>,
    current_idx: usize,
    config: Option<CodingEvalConfig>,
}

impl Default for HumanEvalPlusBenchmark {
    fn default() -> Self {
        Self {
            state: Mutex::new(HumanEvalPlusState {
                items: Vec::new(),
                current_idx: 0,
                config: None,
            }),
        }
    }
}
pub struct MbppPlusBenchmark {
    state: Mutex<MbppPlusState>,
}

struct MbppPlusState {
    items: Vec<serde_json::Value>,
    current_idx: usize,
    config: Option<CodingEvalConfig>,
}

impl Default for MbppPlusBenchmark {
    fn default() -> Self {
        Self {
            state: Mutex::new(MbppPlusState {
                items: Vec::new(),
                current_idx: 0,
                config: None,
            }),
        }
    }
}

const HUMANEVAL_PLUS_URL: &str = "https://github.com/evalplus/humanevalplus_release/releases/download/v0.1.10/HumanEvalPlus.jsonl.gz";
const MBPP_PLUS_URL: &str =
    "https://github.com/evalplus/mbppplus_release/releases/download/v0.2.0/MbppPlus.jsonl.gz";
const HUMAN_EVAL_URL: &str =
    "https://raw.githubusercontent.com/openai/human-eval/6d43fb980f9fee3c892a914eda09951f772ad10d/data/HumanEval.jsonl.gz";
// Python image used to run coding-eval harnesses in a sandboxed container.
const DEFAULT_DOCKER_IMAGE: &str = "python:3.12";

/// Pinned SHA-256 digests for the auto-downloaded coding-eval datasets,
/// computed from the exact bytes served by each pinned URL (fetched 2026-08-04).
/// A mismatch causes the download to fail rather than silently use tampered data.
fn pinned_checksum(task_type: TaskType) -> Option<&'static str> {
    match task_type {
        TaskType::HumanEval => {
            Some("b796127e635a67f93fb35c04f4cb03cf06f38c8072ee7cee8833d7bee06979ef")
        }
        TaskType::HumanEvalPlus => {
            Some("272720b90ac375502c8ed23cd791c2a93dfb22a911641a494da74a426c09f101")
        }
        TaskType::Mbpp => Some("af43697e8791c4c149bdfd6b489d8b5412507551ac20e28a439f650b8225db63"),
    }
}

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
    let sha_path = cache_dir.join(format!("{}.sha256", filename));

    // If a cached copy plus a stored checksum exist, verify before reuse.
    if path.exists() && sha_path.exists() {
        let stored = fs::read_to_string(&sha_path)?.trim().to_string();
        let actual = crate::download::sha256_hex(&fs::read(&path)?);
        if actual.eq_ignore_ascii_case(&stored) {
            return Ok(path);
        }
        // Corrupt cache: remove and re-download below.
        println!(
            "  Coding_eval cache checksum mismatch; re-downloading {}",
            filename
        );
        let _ = fs::remove_file(&path);
        let _ = fs::remove_file(&sha_path);
    } else if path.exists() {
        // Pre-existing cache without a checksum sidecar: accept as-is.
        return Ok(path);
    }

    println!("  Downloading coding_eval taskset {}...", taskset.name);
    let bytes = download_with_retry_bytes_sha256(
        url,
        3,
        60,
        "llm-benchmark-runner",
        pinned_checksum(taskset.task_type),
    )?;
    let tmp_path = path.with_extension(format!(
        "{}.tmp.{}",
        path.extension()
            .and_then(|ext| ext.to_str())
            .unwrap_or("download"),
        std::process::id()
    ));
    fs::write(&tmp_path, &bytes)?;
    fs::rename(&tmp_path, &path).inspect_err(|_rename_err| {
        let _ = fs::remove_file(&tmp_path);
    })?;
    // Cache the checksum alongside so subsequent runs validate the artifact.
    fs::write(&sha_path, crate::download::sha256_hex(&bytes))?;
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
pub fn common_coding_to_report_result(b: &BenchmarkResult) -> Result<BenchmarkResult> {
    let benchmark = CodingEvalBenchmark::default();
    benchmark.to_report_result(b)
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
#[allow(dead_code)] // language and tasks_path reserved for future multi-language support
struct TasksetConfig {
    name: String,
    task_type: TaskType,
    language: String,
    tasks_path: Option<PathBuf>,
}

#[derive(Debug, Clone)]
#[allow(dead_code)] // pass2/pass3/multi-language/host-repo fields reserved for full harness support
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
#[allow(dead_code)] // struct kept for deserialization schema; not all fields consumed
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
    assertion: Option<String>,
}

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

fn load_jsonl(path: &PathBuf) -> Result<Vec<JsonValue>> {
    use flate2::read::GzDecoder;
    use std::io::BufReader;

    let file = fs::File::open(path)?;
    let decoder = GzDecoder::new(BufReader::new(file));
    let rdr = serde_json::Deserializer::from_reader(decoder).into_iter::<JsonValue>();
    let mut items = Vec::new();
    for item in rdr {
        items.push(item?);
    }
    Ok(items)
}

impl Benchmark for CodingEvalBenchmark {
    fn name(&self) -> &str {
        "coding_eval"
    }
    fn display_name(&self) -> &'static str {
        "Coding Eval"
    }
    fn category(&self) -> BenchmarkCategory {
        BenchmarkCategory::ShortContextCoding
    }

    fn pre_execute(&self, config: &yaml_serde::Value) -> Result<()> {
        let cfg = parse_config(config)?;
        let data_path = download_taskset(&cfg.tasksets[0])?;
        let items = load_jsonl(&data_path)?;
        let limit = cfg.num_samples.unwrap_or(items.len());
        println!("Coding Eval: {} problems (limit: {})", items.len(), limit);
        let mut state = self.state.lock().expect(crate::shared::MUTEX_PANIC_MSG);
        state.items = items.into_iter().take(limit).collect();
        state.current_idx = 0;
        state.config = Some(cfg);
        Ok(())
    }

    fn execute_one(
        &self,
        model: &Model,
        _config: &yaml_serde::Value,
        tracker: &mut TokenTracker,
    ) -> Result<Option<TaskResult>> {
        let (idx, item, cfg) = {
            let mut state = self.state.lock().expect(crate::shared::MUTEX_PANIC_MSG);
            if state.current_idx >= state.items.len() {
                return Ok(None);
            }
            let idx = state.current_idx;
            let item = state.items[idx].clone();
            state.current_idx += 1;
            let cfg = state.config.as_ref().expect("config not set").clone();
            (idx, item, cfg)
        };

        let task_name = item
            .get("task_id")
            .and_then(|v| v.as_str())
            .unwrap_or("")
            .to_string();
        let prompt = item
            .get("prompt")
            .and_then(|v| v.as_str())
            .unwrap_or("")
            .to_string();

        let system_prompt = "You are a coding assistant. Generate a complete solution in Python.";
        let response = tracker.chat_completion(&model.model_name, system_prompt, &prompt)?;
        let code = extract_code(&response);
        let passed = run_coding_test(&item, &code, &cfg);

        Ok(Some(
            TaskResult::new(
                format!("task-{}", idx),
                passed,
                if passed { 1.0 } else { 0.0 },
                vec![],
            )
            .with_metadata(Some(
                serde_json::json!({ "task_id": task_name, "correct": passed }),
            )),
        ))
    }

    fn to_report_result(&self, b: &BenchmarkResult) -> Result<BenchmarkResult> {
        let raw = &b.raw;
        let (total, passed, output_tokens, thinking_tokens) = {
            if let Some(per_task) = raw.get("per_task").and_then(|v| v.as_array()) {
                let total = per_task.len() as i64;
                let passed = per_task
                    .iter()
                    .filter(|t| t.get("passed").and_then(|v| v.as_bool()).unwrap_or(false))
                    .count() as i64;
                let out: i64 = per_task
                    .iter()
                    .filter_map(|t| t.get("output_tokens").and_then(|v| v.as_i64()))
                    .sum();
                let think: i64 = per_task
                    .iter()
                    .filter_map(|t| t.get("thinking_tokens").and_then(|v| v.as_i64()))
                    .sum();
                (total, passed, out, think)
            } else {
                (
                    raw.get("total_tasks").and_then(|v| v.as_i64()).unwrap_or(0),
                    raw.get("passed").and_then(|v| v.as_i64()).unwrap_or(0),
                    raw.get("output_tokens")
                        .and_then(|v| v.as_i64())
                        .unwrap_or(0),
                    raw.get("thinking_tokens")
                        .and_then(|v| v.as_i64())
                        .unwrap_or(0),
                )
            }
        };
        let pass_rate = if total > 0 {
            passed as f64 / total as f64 * 100.0
        } else {
            0.0
        };
        let mut scores = BTreeMap::new();
        scores.insert(
            "pass_rate".to_string(),
            Score::float(pass_rate, ScoreUnit::Percent)
                .primary(true)
                .higher_is_better(true),
        );
        scores.insert(
            "passed".to_string(),
            Score::integer(passed, ScoreUnit::Count),
        );
        scores.insert(
            "total_tasks".to_string(),
            Score::integer(total, ScoreUnit::Count),
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
        Ok(BenchmarkResult {
            scores,
            breakdowns: BTreeMap::new(),
            error_classification: BTreeMap::new(),
            artifacts: vec![],
            diagnostics: vec![],
            raw: raw.clone(),
        })
    }
}

impl Benchmark for HumanEvalBenchmark {
    fn name(&self) -> &str {
        "humaneval"
    }
    fn display_name(&self) -> &'static str {
        "HumanEval"
    }
    fn category(&self) -> BenchmarkCategory {
        BenchmarkCategory::ShortContextCoding
    }

    fn pre_execute(&self, config: &yaml_serde::Value) -> Result<()> {
        let preset = preset_config(config, "humaneval", TaskType::HumanEval);
        let cfg = parse_config(&preset)?;
        let taskset = &cfg.tasksets[0];
        let data_path = download_taskset(taskset)?;
        let items = load_jsonl(&data_path)?;
        let limit = cfg.num_samples.unwrap_or(items.len());
        println!("HumanEval: {} problems (limit: {})", items.len(), limit);
        let mut state = self.state.lock().expect(crate::shared::MUTEX_PANIC_MSG);
        state.items = items.into_iter().take(limit).collect();
        state.current_idx = 0;
        state.config = Some(cfg);
        Ok(())
    }

    fn execute_one(
        &self,
        model: &Model,
        _config: &yaml_serde::Value,
        tracker: &mut TokenTracker,
    ) -> Result<Option<TaskResult>> {
        let (idx, item, cfg) = {
            let mut state = self.state.lock().expect(crate::shared::MUTEX_PANIC_MSG);
            if state.current_idx >= state.items.len() {
                return Ok(None);
            }
            let idx = state.current_idx;
            let item = state.items[idx].clone();
            state.current_idx += 1;
            let cfg = state.config.as_ref().expect("config not set").clone();
            (idx, item, cfg)
        };

        let task_name = item
            .get("task_id")
            .and_then(|v| v.as_str())
            .unwrap_or("")
            .to_string();
        let prompt = item
            .get("prompt")
            .and_then(|v| v.as_str())
            .unwrap_or("")
            .to_string();
        let system_prompt = "You are a coding assistant. Generate a complete solution in Python.";
        let response = tracker.chat_completion(&model.model_name, system_prompt, &prompt)?;
        let code = extract_code(&response);
        let passed = run_coding_test(&item, &code, &cfg);

        Ok(Some(
            TaskResult::new(
                format!("task-{}", idx),
                passed,
                if passed { 1.0 } else { 0.0 },
                vec![],
            )
            .with_metadata(Some(
                serde_json::json!({ "task_id": task_name, "correct": passed }),
            )),
        ))
    }

    fn to_report_result(&self, b: &BenchmarkResult) -> Result<BenchmarkResult> {
        common_coding_to_report_result(b)
    }
}

impl Benchmark for HumanEvalPlusBenchmark {
    fn name(&self) -> &str {
        "humaneval_plus"
    }
    fn display_name(&self) -> &'static str {
        "HumanEval+"
    }
    fn category(&self) -> BenchmarkCategory {
        BenchmarkCategory::ShortContextCoding
    }

    fn pre_execute(&self, config: &yaml_serde::Value) -> Result<()> {
        let preset = preset_config(config, "humaneval_plus", TaskType::HumanEvalPlus);
        let cfg = parse_config(&preset)?;
        let taskset = &cfg.tasksets[0];
        let data_path = download_taskset(taskset)?;
        let items = load_jsonl(&data_path)?;
        let limit = cfg.num_samples.unwrap_or(items.len());
        println!("HumanEval+: {} problems (limit: {})", items.len(), limit);
        let mut state = self.state.lock().expect(crate::shared::MUTEX_PANIC_MSG);
        state.items = items.into_iter().take(limit).collect();
        state.current_idx = 0;
        state.config = Some(cfg);
        Ok(())
    }

    fn execute_one(
        &self,
        model: &Model,
        _config: &yaml_serde::Value,
        tracker: &mut TokenTracker,
    ) -> Result<Option<TaskResult>> {
        let (idx, item, cfg) = {
            let mut state = self.state.lock().expect(crate::shared::MUTEX_PANIC_MSG);
            if state.current_idx >= state.items.len() {
                return Ok(None);
            }
            let idx = state.current_idx;
            let item = state.items[idx].clone();
            state.current_idx += 1;
            let cfg = state.config.as_ref().expect("config not set").clone();
            (idx, item, cfg)
        };

        let task_name = item
            .get("task_id")
            .and_then(|v| v.as_str())
            .unwrap_or("")
            .to_string();
        let prompt = item
            .get("prompt")
            .and_then(|v| v.as_str())
            .unwrap_or("")
            .to_string();
        let system_prompt = "You are a coding assistant. Generate a complete solution in Python.";
        let response = tracker.chat_completion(&model.model_name, system_prompt, &prompt)?;
        let code = extract_code(&response);
        let passed = run_coding_test(&item, &code, &cfg);

        Ok(Some(
            TaskResult::new(
                format!("task-{}", idx),
                passed,
                if passed { 1.0 } else { 0.0 },
                vec![],
            )
            .with_metadata(Some(
                serde_json::json!({ "task_id": task_name, "correct": passed }),
            )),
        ))
    }

    fn to_report_result(&self, b: &BenchmarkResult) -> Result<BenchmarkResult> {
        common_coding_to_report_result(b)
    }
}

impl Benchmark for MbppPlusBenchmark {
    fn name(&self) -> &str {
        "mbpp_plus"
    }
    fn display_name(&self) -> &'static str {
        "MBPP+"
    }
    fn category(&self) -> BenchmarkCategory {
        BenchmarkCategory::ShortContextCoding
    }

    fn pre_execute(&self, config: &yaml_serde::Value) -> Result<()> {
        let preset = preset_config(config, "mbpp_plus", TaskType::Mbpp);
        let cfg = parse_config(&preset)?;
        let taskset = &cfg.tasksets[0];
        let data_path = download_taskset(taskset)?;
        let items = load_jsonl(&data_path)?;
        let limit = cfg.num_samples.unwrap_or(items.len());
        println!("MBPP+: {} problems (limit: {})", items.len(), limit);
        let mut state = self.state.lock().expect(crate::shared::MUTEX_PANIC_MSG);
        state.items = items.into_iter().take(limit).collect();
        state.current_idx = 0;
        state.config = Some(cfg);
        Ok(())
    }

    fn execute_one(
        &self,
        model: &Model,
        _config: &yaml_serde::Value,
        tracker: &mut TokenTracker,
    ) -> Result<Option<TaskResult>> {
        let (idx, item, cfg) = {
            let mut state = self.state.lock().expect(crate::shared::MUTEX_PANIC_MSG);
            if state.current_idx >= state.items.len() {
                return Ok(None);
            }
            let idx = state.current_idx;
            let item = state.items[idx].clone();
            state.current_idx += 1;
            let cfg = state.config.as_ref().expect("config not set").clone();
            (idx, item, cfg)
        };

        let task_name = item
            .get("task_id")
            .and_then(|v| v.as_str())
            .unwrap_or("")
            .to_string();
        let prompt = item
            .get("prompt")
            .and_then(|v| v.as_str())
            .unwrap_or("")
            .to_string();
        let system_prompt = "You are a coding assistant. Generate a complete solution in Python.";
        let response = tracker.chat_completion(&model.model_name, system_prompt, &prompt)?;
        let code = extract_code(&response);
        let passed = run_coding_test(&item, &code, &cfg);

        Ok(Some(
            TaskResult::new(
                format!("task-{}", idx),
                passed,
                if passed { 1.0 } else { 0.0 },
                vec![],
            )
            .with_metadata(Some(
                serde_json::json!({ "task_id": task_name, "correct": passed }),
            )),
        ))
    }

    fn to_report_result(&self, b: &BenchmarkResult) -> Result<BenchmarkResult> {
        common_coding_to_report_result(b)
    }
}

fn extract_code(response: &str) -> String {
    // Try to extract code from ```python...``` blocks
    if let Some(start) = response.find("```python") {
        let after = &response[start + 9..];
        if let Some(end) = after.find("```") {
            return after[..end].trim().to_string();
        }
    }
    // Fallback: return entire response
    response.trim().to_string()
}

fn run_coding_test(task_item: &JsonValue, code: &str, cfg: &CodingEvalConfig) -> bool {
    let entry_point = task_item
        .get("entry_point")
        .and_then(|v| v.as_str())
        .unwrap_or("solution")
        .to_string();
    let test_str = task_item
        .get("test")
        .and_then(|v| v.as_str())
        .map(String::from);
    let base_input = task_item
        .get("base_input")
        .and_then(|v| v.as_array())
        .cloned();
    let plus_input = task_item
        .get("plus_input")
        .and_then(|v| v.as_array())
        .cloned();
    let atol = task_item
        .get("atol")
        .and_then(|v| v.as_f64())
        .unwrap_or(1e-6);

    let harness = generate_test_harness(
        &entry_point,
        code,
        &test_str,
        &base_input,
        &plus_input,
        atol,
    );

    let mut file = match tempfile::NamedTempFile::new() {
        Ok(f) => f,
        Err(e) => {
            eprintln!("coding test: failed to create temp file: {}", e);
            return false;
        }
    };
    if let Err(e) = file
        .write_all(harness.as_bytes())
        .and_then(|_| file.flush())
    {
        eprintln!("coding test: failed to write harness: {}", e);
        return false;
    }
    let path = file.path().to_path_buf();
    let filename = match path.file_name().and_then(|s| s.to_str()) {
        Some(s) => s.to_string(),
        None => return false,
    };
    let source_dir = match path.parent() {
        Some(d) => d.to_path_buf(),
        None => return false,
    };

    // Mount the directory holding the harness read-only into the container at
    // /harness (the default /tmp tmpfs is separate) and run it with Python.
    let image = DEFAULT_DOCKER_IMAGE.to_string();
    let mut run_config = DockerRunConfig::new(
        image,
        vec![
            "python3".to_string(),
            "-B".to_string(),
            format!("/harness/{}", filename),
        ],
        cfg.timeout_secs,
    );
    run_config.mounts.push(DockerMount {
        source: source_dir,
        target: "/harness".to_string(),
        readonly: true,
        map_host_repo_path: false,
    });

    match DockerRunner::run(&run_config) {
        Ok(result) => result.success(),
        Err(e) => {
            eprintln!("coding test: docker runner error: {}", e);
            false
        }
    }
}

/// Build a self-contained Python harness embedding the solution plus the
/// task's tests. Prefers the HumanEval-style `test` assertion block when
/// present; otherwise falls back to input/output pairs (EvalPlus-style).
fn generate_test_harness(
    entry_point: &str,
    code: &str,
    test_str: &Option<String>,
    base_input: &Option<Vec<JsonValue>>,
    plus_input: &Option<Vec<JsonValue>>,
    atol: f64,
) -> String {
    let mut h = String::new();
    h.push_str("import sys, traceback\n\n");
    h.push_str(code);
    h.push_str("\n\n");

    if let Some(tests) = test_str {
        // HumanEval-style `test` block (self-contained, references entry_point).
        h.push_str("try:\n");
        for line in tests.lines() {
            h.push_str("    ");
            h.push_str(line);
            h.push('\n');
        }
        h.push_str("    print('TESTS_PASSED')\n");
        h.push_str("    sys.exit(0)\n");
        h.push_str("except Exception:\n");
        h.push_str("    traceback.print_exc()\n");
        h.push_str("    print('TESTS_FAILED')\n");
        h.push_str("    sys.exit(1)\n");
        return h;
    }

    // EvalPlus-style: iterate base_input / plus_input against entry_point.
    let mut inputs: Vec<JsonValue> = Vec::new();
    if let Some(bi) = base_input {
        inputs.extend(bi.iter().cloned());
    }
    if let Some(pi) = plus_input {
        inputs.extend(pi.iter().cloned());
    }
    let inputs_json = serde_json::to_string(&inputs).unwrap_or_else(|_| "[]".to_string());
    h.push_str(&format!("ATOL = {}\n", atol));
    h.push_str(&format!("INPUTS = {}\n", inputs_json));
    h.push_str("def run() -> int:\n");
    h.push_str("    for i, tc in enumerate(INPUTS):\n");
    h.push_str("        try:\n");
    h.push_str(&format!(
        "            out = {}(**tc.get('input', {{}}))\n",
        entry_point
    ));
    h.push_str("            exp = tc.get('output')\n");
    h.push_str("            if exp is not None:\n");
    h.push_str("                if isinstance(out, float) and isinstance(exp, float):\n");
    h.push_str("                    if abs(out - exp) <= ATOL:\n");
    h.push_str("                        continue\n");
    h.push_str("                if out == exp:\n");
    h.push_str("                    continue\n");
    h.push_str("            print('FAIL test', i)\n");
    h.push_str("            return 1\n");
    h.push_str("        except Exception as e:\n");
    h.push_str("            print('ERR test', i, repr(e))\n");
    h.push_str("            return 1\n");
    h.push_str("    print('TESTS_PASSED')\n");
    h.push_str("    return 0\n");
    h.push_str("sys.exit(run())\n");
    h
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn harness_prefers_test_block_for_humaneval() {
        let h = generate_test_harness(
            "my_func",
            "def my_func(x): return x + 1",
            &Some("assert my_func(1) == 2".to_string()),
            &None,
            &None,
            1e-6,
        );
        assert!(h.contains("def my_func(x): return x + 1"));
        assert!(h.contains("assert my_func(1) == 2"));
        assert!(h.contains("TESTS_PASSED"));
    }

    #[test]
    fn harness_generates_input_loop_for_evalplus() {
        let inputs = vec![json!({"input": {"x": 1}, "output": 2})];
        let h = generate_test_harness(
            "my_func",
            "def my_func(x): return x + 1",
            &None,
            &Some(inputs),
            &None,
            1e-6,
        );
        assert!(h.contains("my_func(**tc.get('input', {}))"));
        assert!(h.contains("INPUTS"));
    }

    #[test]
    fn harness_falls_back_to_plus_input() {
        let plus = vec![json!({"input": {}, "output": 3})];
        let h = generate_test_harness("f", "def f(): return 3", &None, &None, &Some(plus), 1e-6);
        assert!(h.contains("INPUTS"));
    }
}
