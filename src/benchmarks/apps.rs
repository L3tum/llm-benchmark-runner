//! APPS Benchmark — Coding problems with automated test evaluation.
//!
//! Based on the APPS dataset (codeparrot/apps on HuggingFace).
//! Two problem types:
//! - Call-based: Write a function called `solve()`
//! - Standard Input: Read from stdin, write to stdout
//!
//! Evaluation runs Python tests in a Docker container.

use crate::benchmarks::Benchmark;
use crate::config::Model;
use crate::docker_runner::{DockerRunConfig, DockerRunner};
use crate::shared::{
    BenchmarkCategory, BenchmarkResult, BreakdownTable, Score, ScoreUnit, TaskResult,
};
use crate::token_tracker::TokenTracker;
use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use serde_json::Value as JsonValue;
use std::collections::{BTreeMap, HashMap};
use std::fs;
use std::io::Write;
use std::path::PathBuf;
use std::sync::Mutex;

pub struct AppsBenchmark {
    state: Mutex<AppsState>,
}

struct AppsState {
    items: Vec<AppsInstance>,
    current_idx: usize,
    config: AppsConfig,
    /// Generated solutions per instance
    generated_solutions: HashMap<String, AppsSolution>,
}

impl Default for AppsBenchmark {
    fn default() -> Self {
        Self {
            state: Mutex::new(AppsState {
                items: Vec::new(),
                current_idx: 0,
                config: AppsConfig::default(),
                generated_solutions: HashMap::new(),
            }),
        }
    }
}

#[derive(Debug, Clone, Deserialize)]
struct AppsConfig {
    #[serde(default)]
    num_samples: Option<usize>,
    #[serde(default)]
    difficulty_filter: Option<Vec<String>>,
    #[serde(default = "default_timeout")]
    timeout_secs: u64,
    #[serde(default = "default_docker_image")]
    docker_image: String,
}

fn default_timeout() -> u64 {
    120
}

fn default_docker_image() -> String {
    "llm-benchmark-runner-apps-eval:latest".to_string()
}

impl Default for AppsConfig {
    fn default() -> Self {
        Self {
            num_samples: None,
            difficulty_filter: None,
            timeout_secs: default_timeout(),
            docker_image: default_docker_image(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct AppsInstance {
    #[serde(rename = "problem_id")]
    problem_id: String,
    problem: String,
    #[serde(default)]
    input_output: JsonValue,
    #[serde(default)]
    starter_code: String,
    #[serde(default, rename = "is_unit_test")]
    is_unit_test: String, // "yes" or "no"
    #[serde(default)]
    difficulty: String,
    #[serde(default)]
    test: String,
    #[serde(default, rename = "output")]
    expected_output: String,
    #[serde(default, rename = "input")]
    input_text: String,
    #[serde(default)]
    test_type: String,
    #[serde(default)]
    start_code: String,
}

impl AppsInstance {
    fn is_call_based(&self) -> bool {
        self.is_unit_test == "yes" || !self.test.is_empty()
    }
}

#[derive(Debug, Clone)]
struct AppsSolution {
    code: String,
    is_call_based: bool,
    task_idx: usize,
}

const APPS_HF_DATASET: &str = "codeparrot/apps";

const SYSTEM_PROMPT: &str =
    "You are a coding assistant. Write Python code to solve the problem. Output your code in a ```python code block.";

fn load_apps_dataset(config: &AppsConfig) -> Result<Vec<AppsInstance>> {
    let cache_dir = dirs::cache_dir()
        .unwrap_or_default()
        .join("llm-benchmark-runner")
        .join("apps");
    let path = cache_dir.join("apps_train.jsonl");

    if path.exists() {
        let content = fs::read_to_string(&path)?;
        let items: Vec<AppsInstance> = content
            .lines()
            .filter(|l| !l.trim().is_empty())
            .map(|line| {
                serde_json::from_str(line).context(format!(
                    "failed to parse APPS instance: {}",
                    &line[..line.len().min(50)]
                ))
            })
            .collect::<Result<Vec<_>>>()?;
        return apply_filters(items, config);
    }

    // Download from HuggingFace rows API
    fs::create_dir_all(&cache_dir)?;
    println!("  Downloading APPS dataset from HuggingFace...");

    let client = reqwest::blocking::Client::builder().build()?;
    let mut all_rows = Vec::new();
    let mut offset = 0usize;
    let page_size = 500usize;

    loop {
        let url = reqwest::Url::parse_with_params(
            "https://datasets-server.huggingface.co/rows",
            &[
                ("dataset", APPS_HF_DATASET),
                ("config", "default"),
                ("split", "train"),
                ("offset", &offset.to_string()),
                ("length", &page_size.to_string()),
            ],
        )?;

        let response: JsonValue = client.get(url).send()?.error_for_status()?.json()?;
        let page = response
            .get("rows")
            .and_then(|v| v.as_array())
            .cloned()
            .unwrap_or_default();

        if page.is_empty() {
            break;
        }

        for item in &page {
            let row = item
                .get("row")
                .cloned()
                .ok_or_else(|| anyhow::anyhow!("missing row in HuggingFace response"))?;
            let instance: AppsInstance = serde_json::from_value(row)?;
            all_rows.push(instance);
        }

        if page.len() < page_size {
            break;
        }
        offset += page_size;
    }

    // Cache to JSONL
    let tmp_path = path.with_extension(format!("jsonl.tmp.{}", std::process::id()));
    {
        let mut file = fs::File::create(&tmp_path)?;
        for item in &all_rows {
            writeln!(file, "{}", serde_json::to_string(item)?)?;
        }
    }
    fs::rename(&tmp_path, &path)?;

    apply_filters(all_rows, config)
}

fn apply_filters(items: Vec<AppsInstance>, config: &AppsConfig) -> Result<Vec<AppsInstance>> {
    let mut filtered = items;

    if let Some(ref difficulties) = config.difficulty_filter {
        filtered.retain(|i| difficulties.contains(&i.difficulty));
    }

    if let Some(limit) = config.num_samples {
        filtered.truncate(limit);
    }

    Ok(filtered)
}

fn extract_code(response: &str) -> String {
    // Try to extract code from ```python...``` blocks
    if let Some(start) = response.find("```python") {
        let after = &response[start + 9..];
        if let Some(end) = after.find("```") {
            return after[..end].trim().to_string();
        }
    }
    // Try ```code...``` (language unspecified)
    if let Some(start) = response.find("```") {
        let after = &response[start + 3..];
        if let Some(end) = after.find("```") {
            let code = after[..end].trim();
            // Check if it looks like Python
            if code.contains("def ") || code.contains("import ") || code.contains("print(") {
                return code.to_string();
            }
        }
    }
    // Fallback: return entire response
    response.trim().to_string()
}

fn build_prompt(instance: &AppsInstance) -> String {
    let intro = if instance.is_call_based() {
        "Write a Python function called `solve()` that solves the following problem.\n"
    } else {
        "Write a Python program that reads from stdin and writes to stdout to solve the following problem.\n"
    };

    let starter = if !instance.starter_code.is_empty() {
        format!("\n\nStarter code:\n```\n{}\n```\n\n", instance.starter_code)
    } else if !instance.start_code.is_empty() {
        format!("\n\nStarter code:\n```\n{}\n```\n\n", instance.start_code)
    } else {
        String::new()
    };

    format!("{}Problem:\n{}\n{}", intro, instance.problem, starter)
}

fn ensure_apps_docker_image(cfg: &AppsConfig) -> Result<()> {
    // Check if image exists
    if let Ok(true) = DockerRunner::image_exists(&cfg.docker_image) {
        return Ok(());
    }

    // Build from Dockerfile
    let dockerfile_path = PathBuf::from("docker/apps-eval/Dockerfile");
    if dockerfile_path.exists() {
        println!("  Building APPS evaluation Docker image...");
        let build_cfg = crate::docker_runner::DockerBuildConfig {
            image: cfg.docker_image.clone(),
            dockerfile: dockerfile_path.clone(),
            context: dockerfile_path.parent().unwrap().to_path_buf(),
            timeout_secs: 300,
            host_repo_path: None,
        };
        DockerRunner::build_image(&build_cfg)?;
    } else {
        // Fall back to python:3.12-slim
        eprintln!("  WARNING: docker/apps-eval/Dockerfile not found, using python:3.12-slim");
    }

    Ok(())
}

fn build_evaluation_script() -> Result<String> {
    let mut script = String::new();
    script.push_str(
        r#"import json, sys, subprocess, traceback

def evaluate_call_based(solution_code: str, test_code: str) -> bool:
    try:
        namespace = {}
        exec(solution_code, namespace)
        exec(test_code, namespace)
        return True
    except Exception:
        return False

def evaluate_stdin(solution_code: str, input_text: str, expected_output: str) -> bool:
    try:
        proc = subprocess.run(
            [sys.executable, "-c", solution_code],
            input=input_text, capture_output=True, text=True, timeout=10
        )
        if proc.returncode != 0:
            return False
        return proc.stdout.strip() == expected_output.strip()
    except Exception:
        return False

def main():
    with open("/work/problems.json") as f:
        problems = json.load(f)
    results = {}
    for p in problems:
        task_id = p["task_id"]
        try:
            if p["is_call_based"] and p.get("test"):
                results[task_id] = evaluate_call_based(p["solution_code"], p["test"])
            elif not p["is_call_based"] and p.get("input"):
                results[task_id] = evaluate_stdin(p["solution_code"], p["input"], p["expected_output"])
            else:
                results[task_id] = False
        except Exception as e:
            results[task_id] = False
    with open("/work/results.json", "w") as f:
        json.dump(results, f)

if __name__ == "__main__":
    main()
"#,
    );

    Ok(script)
}

impl Benchmark for AppsBenchmark {
    fn name(&self) -> &str {
        "apps"
    }

    fn display_name(&self) -> &'static str {
        "APPS"
    }

    fn category(&self) -> BenchmarkCategory {
        BenchmarkCategory::ShortContextCoding
    }

    fn pre_execute(&self, config: &yaml_serde::Value) -> Result<()> {
        let cfg = if let Some(raw) = config.get("apps") {
            serde_json::from_value(serde_json::to_value(raw).unwrap_or_default())
                .unwrap_or(AppsConfig::default())
        } else {
            serde_json::from_value(serde_json::to_value(config).unwrap_or_default())
                .unwrap_or(AppsConfig::default())
        };

        ensure_apps_docker_image(&cfg)?;
        let items = load_apps_dataset(&cfg)?;
        println!(
            "  APPS: {} problems loaded ({} call-based, {} stdin)",
            items.len(),
            items.iter().filter(|i| i.is_call_based()).count(),
            items.iter().filter(|i| !i.is_call_based()).count()
        );

        let mut state = self.state.lock().expect(crate::shared::MUTEX_PANIC_MSG);
        state.items = items;
        state.current_idx = 0;
        state.config = cfg;
        Ok(())
    }

    fn execute_one(
        &self,
        model: &Model,
        _config: &yaml_serde::Value,
        tracker: &mut TokenTracker,
    ) -> Result<Option<TaskResult>> {
        let (instance, idx) = {
            let mut state = self.state.lock().expect(crate::shared::MUTEX_PANIC_MSG);
            if state.current_idx >= state.items.len() {
                return Ok(None);
            }
            let idx = state.current_idx;
            let instance = state.items[idx].clone();
            state.current_idx += 1;
            (instance, idx)
        };

        let prompt = build_prompt(&instance);
        let response = tracker.chat_completion(&model.model_name, SYSTEM_PROMPT, &prompt)?;

        let code = extract_code(&response);
        let is_call_based = instance.is_call_based();

        // Store solution for batch evaluation
        {
            let mut state = self.state.lock().expect(crate::shared::MUTEX_PANIC_MSG);
            state.generated_solutions.insert(
                format!("task-{}", idx),
                AppsSolution {
                    code,
                    is_call_based,
                    task_idx: idx,
                },
            );
        }

        Ok(Some(
            TaskResult::new(
                format!("task-{}", idx),
                false, // Will be updated in batch_evaluate
                0.0,
                vec![instance.difficulty.clone()],
            )
            .with_metadata(Some(serde_json::json!({
                "problem_id": instance.problem_id,
                "difficulty": instance.difficulty,
                "is_call_based": is_call_based,
            }))),
        ))
    }

    fn batch_evaluate(
        &self,
        task_results: &[TaskResult],
        _config: &yaml_serde::Value,
    ) -> Result<Option<Vec<TaskResult>>> {
        let (solutions, instances, cfg) = {
            let state = self.state.lock().expect(crate::shared::MUTEX_PANIC_MSG);
            if state.generated_solutions.is_empty() {
                return Ok(None);
            }
            (
                state.generated_solutions.clone(),
                state.items.clone(),
                state.config.clone(),
            )
        };

        // Build problems list for evaluation
        let mut problems: Vec<serde_json::Map<String, JsonValue>> = Vec::new();

        for tr in task_results {
            let task_id = &tr.task_id;
            if let Some(solution) = solutions.get(task_id) {
                if let Some(instance) = instances.get(solution.task_idx) {
                    let mut problem = serde_json::Map::new();
                    problem.insert("task_id".to_string(), JsonValue::String(task_id.clone()));
                    problem.insert(
                        "solution_code".to_string(),
                        JsonValue::String(solution.code.clone()),
                    );
                    problem.insert(
                        "is_call_based".to_string(),
                        JsonValue::Bool(solution.is_call_based),
                    );
                    if solution.is_call_based && !instance.test.is_empty() {
                        problem
                            .insert("test".to_string(), JsonValue::String(instance.test.clone()));
                    }
                    if !solution.is_call_based && !instance.input_text.is_empty() {
                        problem.insert(
                            "input".to_string(),
                            JsonValue::String(instance.input_text.clone()),
                        );
                        problem.insert(
                            "expected_output".to_string(),
                            JsonValue::String(instance.expected_output.clone()),
                        );
                    }
                    problems.push(problem);
                }
            }
        }

        if problems.is_empty() {
            return Ok(None);
        }

        // Create temp directory
        let run_tempdir = tempfile::tempdir_in(
            dirs::cache_dir()
                .unwrap_or_else(|| PathBuf::from("/tmp"))
                .join("llm-benchmark-runner"),
        )
        .context("failed to create temp dir for APPS evaluation")?;
        let run_dir = run_tempdir.path();

        // Write problems and evaluation script
        fs::write(
            run_dir.join("problems.json"),
            serde_json::to_string_pretty(&problems)?,
        )?;
        let eval_script = build_evaluation_script()?;
        fs::write(run_dir.join("evaluate.py"), &eval_script)?;

        // Run evaluation in Docker
        println!(
            "  Running APPS evaluation for {} problems in Docker...",
            problems.len()
        );

        let mut docker = DockerRunConfig::new(
            &cfg.docker_image,
            vec!["python".to_string(), "evaluate.py".to_string()],
            cfg.timeout_secs,
        );
        docker
            .mounts
            .push(crate::docker_runner::DockerMount::readwrite(
                run_dir, "/work",
            ));
        docker.workdir = Some("/work".to_string());
        docker.name_prefix = "llm-benchmark-runner-apps".to_string();
        docker.network_none = true;
        docker.read_only_root = true;
        docker.tmpfs = vec!["/tmp".to_string()];
        docker.pids_limit = Some(50);

        let output = DockerRunner::run(&docker)?;
        println!("  Evaluation complete. Exit code: {:?}", output.exit_code);

        // Parse results
        let results_path = run_dir.join("results.json");
        let mut passed_map: HashMap<String, bool> = HashMap::new();

        if results_path.exists() {
            let results: HashMap<String, bool> =
                serde_json::from_str(&fs::read_to_string(&results_path)?)?;
            passed_map = results;
        }

        // Build new task results
        let mut new_results = Vec::new();
        for tr in task_results {
            let mut new_tr = tr.clone();
            let passed = passed_map.get(&tr.task_id).copied().unwrap_or(false);
            new_tr.passed = passed;
            new_tr.score = if passed { 1.0 } else { 0.0 };
            new_results.push(new_tr);
        }

        Ok(Some(new_results))
    }

    fn to_report_result(&self, b: &BenchmarkResult) -> Result<BenchmarkResult> {
        let raw = &b.raw;

        let per_task: Vec<&serde_json::Value> = raw
            .get("per_task")
            .and_then(|v| v.as_array())
            .map(|arr| arr.iter().collect())
            .unwrap_or_default();

        let total = per_task.len();
        let mut total_score = 0.0;

        // Per-difficulty breakdown
        let mut difficulty_scores: BTreeMap<String, Vec<f64>> = BTreeMap::new();

        for task in &per_task {
            let meta = task.get("metadata").unwrap_or(&JsonValue::Null);
            let passed = task
                .get("passed")
                .and_then(|v| v.as_bool())
                .unwrap_or(false);
            let score = if passed { 1.0 } else { 0.0 };
            total_score += score;

            if let Some(difficulty) = meta.get("difficulty").and_then(|v| v.as_str()) {
                difficulty_scores
                    .entry(difficulty.to_string())
                    .or_default()
                    .push(score);
            }
        }

        let pass_rate = if total > 0 {
            total_score / total as f64 * 100.0
        } else {
            0.0
        };

        let mut scores = BTreeMap::new();
        scores.insert(
            "pass_rate".to_string(),
            Score::float(pass_rate, ScoreUnit::Percent)
                .primary(true)
                .higher_is_better(true)
                .display(format!("{:.1}%", pass_rate)),
        );
        scores.insert(
            "total_problems".to_string(),
            Score::integer(total as i64, ScoreUnit::Count),
        );

        // Per-difficulty breakdown
        let mut difficulty_rows = BTreeMap::new();
        for (difficulty, scores_list) in &difficulty_scores {
            let mean = scores_list.iter().sum::<f64>() / scores_list.len() as f64;
            difficulty_rows.insert(
                difficulty.clone(),
                BTreeMap::from_iter([
                    (
                        "pass_rate".to_string(),
                        Score::float(mean * 100.0, ScoreUnit::Percent)
                            .display(format!("{:.1}%", mean * 100.0)),
                    ),
                    (
                        "count".to_string(),
                        Score::integer(scores_list.len() as i64, ScoreUnit::Count),
                    ),
                ]),
            );
        }

        let mut breakdowns = BTreeMap::new();
        if !difficulty_rows.is_empty() {
            breakdowns.insert(
                "Per-Difficulty Breakdown".to_string(),
                BreakdownTable {
                    title: "Pass Rate by Difficulty".to_string(),
                    rows: difficulty_rows,
                },
            );
        }

        Ok(BenchmarkResult {
            scores,
            breakdowns,
            error_classification: BTreeMap::new(),
            artifacts: vec![],
            diagnostics: vec![crate::shared::Diagnostic {
                level: "info".to_string(),
                message: format!(
                    "APPS: {:.1}% pass rate across {} problems",
                    pass_rate, total
                ),
            }],
            raw: raw.clone(),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn extract_python_code_from_fence() {
        let response = "Here's the solution:\n```python\ndef solve(x):\n    return x + 1\n```";
        let code = extract_code(response);
        assert!(code.contains("def solve"));
        assert!(!code.contains("```"));
    }

    #[test]
    fn extract_python_code_from_generic_fence() {
        let response = "Here's the solution:\n```\ndef solve(x):\n    return x + 1\n```";
        let code = extract_code(response);
        assert!(code.contains("def solve"));
    }

    #[test]
    fn extract_code_fallback_no_fence() {
        let response = "def solve(x):\n    return x + 1";
        let code = extract_code(response);
        assert_eq!(code, "def solve(x):\n    return x + 1");
    }

    #[test]
    fn apps_instance_deserializes_correctly() {
        let json = r#"{"problem_id": "1", "problem": "test", "is_unit_test": "yes", "difficulty": "introductory", "test": "assert solve(1) == 2"}"#;
        let instance: AppsInstance = serde_json::from_str(json).unwrap();
        assert!(instance.is_call_based());
        assert_eq!(instance.difficulty, "introductory");
    }

    #[test]
    fn apps_instance_stdin_mode() {
        let json = r#"{"problem_id": "2", "problem": "test", "is_unit_test": "no", "difficulty": "interview", "input": "5", "output": "10"}"#;
        let instance: AppsInstance = serde_json::from_str(json).unwrap();
        assert!(!instance.is_call_based());
        assert_eq!(instance.input_text, "5");
        assert_eq!(instance.expected_output, "10");
    }

    #[test]
    fn build_prompt_call_based() {
        let instance = AppsInstance {
            problem_id: "1".to_string(),
            problem: "Write a function to add two numbers.".to_string(),
            input_output: JsonValue::Null,
            starter_code: String::new(),
            is_unit_test: "yes".to_string(),
            difficulty: "introductory".to_string(),
            test: "assert solve(1, 2) == 3".to_string(),
            expected_output: String::new(),
            input_text: String::new(),
            test_type: String::new(),
            start_code: String::new(),
        };
        let prompt = build_prompt(&instance);
        assert!(prompt.contains("solve()"));
        assert!(prompt.contains("Write a function to add two numbers"));
    }

    #[test]
    fn build_prompt_stdin() {
        let instance = AppsInstance {
            problem_id: "2".to_string(),
            problem: "Read a number and double it.".to_string(),
            input_output: JsonValue::Null,
            starter_code: String::new(),
            is_unit_test: "no".to_string(),
            difficulty: "interview".to_string(),
            test: String::new(),
            expected_output: "10".to_string(),
            input_text: "5".to_string(),
            test_type: String::new(),
            start_code: String::new(),
        };
        let prompt = build_prompt(&instance);
        assert!(prompt.contains("stdin"));
        assert!(prompt.contains("Read a number and double it"));
    }

    #[test]
    fn difficulty_filtering() {
        let items = vec![
            AppsInstance {
                problem_id: "1".to_string(),
                problem: "test".to_string(),
                input_output: JsonValue::Null,
                starter_code: String::new(),
                is_unit_test: "yes".to_string(),
                difficulty: "introductory".to_string(),
                test: String::new(),
                expected_output: String::new(),
                input_text: String::new(),
                test_type: String::new(),
                start_code: String::new(),
            },
            AppsInstance {
                problem_id: "2".to_string(),
                problem: "test".to_string(),
                input_output: JsonValue::Null,
                starter_code: String::new(),
                is_unit_test: "yes".to_string(),
                difficulty: "interview".to_string(),
                test: String::new(),
                expected_output: String::new(),
                input_text: String::new(),
                test_type: String::new(),
                start_code: String::new(),
            },
        ];
        let config = AppsConfig {
            num_samples: None,
            difficulty_filter: Some(vec!["introductory".to_string()]),
            timeout_secs: 120,
            docker_image: "test".to_string(),
        };
        let filtered = apply_filters(items, &config).unwrap();
        assert_eq!(filtered.len(), 1);
        assert_eq!(filtered[0].difficulty, "introductory");
    }
}
