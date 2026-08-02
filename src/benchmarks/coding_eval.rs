use crate::benchmarks::Benchmark;
use crate::config::Model;
use crate::download::download_with_retry;
use crate::shared::{BenchmarkCategory, BenchmarkResult, Score, ScoreUnit, TaskResult};
use crate::token_tracker::TokenTracker;
use anyhow::Result;
use serde::Deserialize;
use serde_json::Value as JsonValue;
use std::collections::{BTreeMap, HashMap};
use std::fs;
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
    "https://github.com/openai/human-eval/raw/master/data/HumanEval.jsonl.gz";
#[allow(dead_code)] // used for Docker-based coding benchmarks
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
        let passed = run_tests_simple(&prompt, &code, &cfg);

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
        let passed = run_tests_simple(&prompt, &code, &cfg);

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
        let passed = run_tests_simple(&prompt, &code, &cfg);

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
        let passed = run_tests_simple(&prompt, &code, &cfg);

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

fn run_tests_simple(_prompt: &str, _code: &str, _cfg: &CodingEvalConfig) -> bool {
    // Simplified test runner - in production this would spawn docker/python
    false
}
