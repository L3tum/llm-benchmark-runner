use crate::client::Client;
use crate::config::Model;
use anyhow::{Context, Result};
use flate2::read::GzDecoder;
use regex::Regex;
use serde::Deserialize;
use serde_json::Value as JsonValue;
use std::collections::HashMap;
use std::ffi::OsStr;
use std::fs;
use std::io::Read;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

pub struct CodingEvalBenchmark;

const HUMANEVAL_PLUS_URL: &str = "https://github.com/evalplus/humanevalplus_release/releases/download/v0.1.10/HumanEvalPlus.jsonl.gz";
const MBPP_PLUS_URL: &str =
    "https://github.com/evalplus/mbppplus_release/releases/download/v0.2.0/MbppPlus.jsonl.gz";
const HUMAN_EVAL_URL: &str =
    "https://github.com/openai/human-eval/raw/master/data/HumanEval.jsonl.gz";
const DEFAULT_DOCKER_IMAGE: &str = "python:3.12";

#[derive(Debug, Clone, PartialEq, Eq)]
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

    fn pre_execute(&self, config: &serde_yaml::Value) -> Result<()> {
        let parsed = parse_config(config)?;
        for taskset in &parsed.tasksets {
            if taskset.tasks_path.is_none() {
                download_taskset(taskset)?;
            }
        }
        Ok(())
    }

    fn execute(&self, model: &Model, config: &serde_yaml::Value) -> Result<serde_json::Value> {
        let cfg = parse_config(config)?;
        let client = Client::new(&model.proxy)?;
        let max_attempts = if cfg.enable_pass3 {
            3
        } else if cfg.enable_pass2 {
            2
        } else {
            1
        };

        let mut all_task_records = Vec::new();
        let mut by_taskset: HashMap<String, TasksetStats> = HashMap::new();
        let mut total_output_tokens = 0u64;
        let mut total_thinking_tokens = 0u64;
        let mut total_tasks = 0usize;
        let mut pass_counts = [0usize; 3];
        let mut timeout_count = 0usize;
        let mut skipped_later_attempts = 0usize;

        for taskset in &cfg.tasksets {
            if taskset.language != "python" {
                return Err(anyhow::anyhow!(
                    "coding_eval language runner not implemented: {} (taskset {})",
                    taskset.language,
                    taskset.name
                ));
            }
            let docker_image = cfg
                .language_images
                .get(&taskset.language)
                .cloned()
                .unwrap_or_else(|| DEFAULT_DOCKER_IMAGE.to_string());
            let mut tasks = load_tasks(taskset)?;
            if let Some(limit) = cfg.num_samples {
                tasks.truncate(limit);
            }
            println!(
                "\nEvaluating coding_eval {}: {} tasks ({})",
                taskset.name,
                tasks.len(),
                taskset.language
            );

            let mut stats = TasksetStats::new(max_attempts);
            for task in &tasks {
                total_tasks += 1;
                stats.total += 1;
                let mut attempts = Vec::new();
                let mut previous_solution: Option<String> = None;
                let mut previous_result: Option<DockerResult> = None;
                let mut passed_attempt: Option<usize> = None;

                for attempt in 1..=max_attempts {
                    if let Some(first_pass) = passed_attempt {
                        skipped_later_attempts += 1;
                        stats.skipped_later_attempts += 1;
                        attempts.push(AttemptOutcome {
                            attempt,
                            skipped: true,
                            passed: true,
                            timed_out: false,
                            exit_code: Some(0),
                            stdout: String::new(),
                            stderr: String::new(),
                            error_summary: format!(
                                "Skipped: task already passed at attempt {}",
                                first_pass
                            ),
                            output_tokens: 0,
                            thinking_tokens: 0,
                        });
                        continue;
                    }

                    let prompt = if attempt == 1 {
                        build_initial_prompt(task, &taskset.language)
                    } else {
                        build_repair_prompt(
                            task,
                            &taskset.language,
                            previous_solution.as_deref().unwrap_or_default(),
                            previous_result.as_ref(),
                        )
                    };
                    let (response, output_tokens, thinking_tokens) =
                        client.chat_completion(&model.model_name, "", &prompt)?;
                    let output_tokens = output_tokens.unwrap_or(0);
                    let thinking_tokens = thinking_tokens.unwrap_or(0);
                    total_output_tokens += output_tokens;
                    total_thinking_tokens += thinking_tokens;

                    let solution = extract_code(&response);
                    let run_dir =
                        run_directory(&model.display_name, &taskset.name, &task.task_id, attempt);
                    let harness = write_harness(&run_dir, task, &taskset.task_type, &solution)
                        .with_context(|| format!("failed to write harness for {}", task.task_id))?;
                    let docker_result = run_in_docker(
                        &harness.test_path,
                        &run_dir,
                        &docker_image,
                        cfg.timeout_secs,
                        cfg.host_repo_path.as_deref(),
                    );
                    let docker_result = match docker_result {
                        Ok(result) => result,
                        Err(e) => DockerResult {
                            passed: false,
                            timed_out: false,
                            exit_code: None,
                            stdout: String::new(),
                            stderr: String::new(),
                            error_summary: e.to_string(),
                        },
                    };
                    if docker_result.timed_out {
                        timeout_count += 1;
                        stats.timeout_count += 1;
                    }
                    if docker_result.passed {
                        passed_attempt = Some(attempt);
                    }
                    previous_solution = Some(solution);
                    previous_result = Some(docker_result.clone());
                    attempts.push(AttemptOutcome {
                        attempt,
                        skipped: false,
                        passed: docker_result.passed,
                        timed_out: docker_result.timed_out,
                        exit_code: docker_result.exit_code,
                        stdout: docker_result.stdout,
                        stderr: docker_result.stderr,
                        error_summary: docker_result.error_summary,
                        output_tokens,
                        thinking_tokens,
                    });
                }

                for k in 1..=max_attempts {
                    if passed_attempt.is_some_and(|attempt| attempt <= k) {
                        pass_counts[k - 1] += 1;
                        stats.pass_counts[k - 1] += 1;
                    }
                }

                all_task_records.push(serde_json::json!({
                    "taskset": taskset.name,
                    "task_type": taskset.task_type.as_str(),
                    "language": taskset.language,
                    "task_id": task.task_id,
                    "entry_point": task.entry_point,
                    "passed": passed_attempt.is_some(),
                    "passed_attempt": passed_attempt,
                    "attempts": attempts_to_json(&attempts),
                }));
            }
            by_taskset.insert(taskset.name.clone(), stats);
        }

        let mut result = serde_json::Map::new();
        result.insert(
            "pass_at_1".to_string(),
            serde_json::json!(rate(pass_counts[0], total_tasks)),
        );
        if max_attempts >= 2 {
            result.insert(
                "pass_at_2".to_string(),
                serde_json::json!(rate(pass_counts[1], total_tasks)),
            );
        }
        if max_attempts >= 3 {
            result.insert(
                "pass_at_3".to_string(),
                serde_json::json!(rate(pass_counts[2], total_tasks)),
            );
        }
        result.insert(
            "passed".to_string(),
            serde_json::json!(pass_counts[max_attempts - 1]),
        );
        result.insert(
            "total_questions".to_string(),
            serde_json::json!(total_tasks),
        );
        result.insert(
            "timeout_count".to_string(),
            serde_json::json!(timeout_count),
        );
        result.insert(
            "skipped_later_attempts".to_string(),
            serde_json::json!(skipped_later_attempts),
        );
        result.insert(
            "output_tokens".to_string(),
            serde_json::json!(total_output_tokens),
        );
        result.insert(
            "thinking_tokens".to_string(),
            serde_json::json!(total_thinking_tokens),
        );
        result.insert(
            "results_by_taskset".to_string(),
            serde_json::json!(taskset_stats_to_json(&by_taskset, max_attempts)),
        );
        result.insert("tasks".to_string(), serde_json::json!(all_task_records));
        Ok(serde_json::Value::Object(result))
    }
}

#[derive(Debug, Clone)]
struct TasksetStats {
    total: usize,
    pass_counts: [usize; 3],
    timeout_count: usize,
    skipped_later_attempts: usize,
}

impl TasksetStats {
    fn new(_max_attempts: usize) -> Self {
        Self {
            total: 0,
            pass_counts: [0; 3],
            timeout_count: 0,
            skipped_later_attempts: 0,
        }
    }
}

fn parse_config(config: &serde_yaml::Value) -> Result<CodingEvalConfig> {
    let num_samples = config
        .get("num_samples")
        .and_then(|v| v.as_i64())
        .map(|n| n.max(0) as usize);
    let timeout_secs = config
        .get("timeout_secs")
        .and_then(|v| v.as_i64())
        .unwrap_or(8)
        .max(1) as u64;
    let enable_pass3 = config
        .get("enable_pass3")
        .and_then(|v| v.as_bool())
        .unwrap_or(false);
    let enable_pass2 = enable_pass3
        || config
            .get("enable_pass2")
            .and_then(|v| v.as_bool())
            .unwrap_or(false);

    let mut language_images = HashMap::new();
    language_images.insert("python".to_string(), DEFAULT_DOCKER_IMAGE.to_string());
    if let Some(map) = config.get("language_images").and_then(|v| v.as_mapping()) {
        for (k, v) in map {
            if let (Some(lang), Some(image)) = (k.as_str(), v.as_str()) {
                language_images.insert(lang.to_string(), image.to_string());
            }
        }
    }

    let host_repo_path = config
        .get("host_repo_path")
        .and_then(|v| v.as_str())
        .map(PathBuf::from);

    let tasksets = if let Some(value) = config.get("tasksets") {
        parse_tasksets(value)?
    } else {
        vec![TasksetConfig {
            name: "humaneval_plus".to_string(),
            task_type: TaskType::HumanEvalPlus,
            language: "python".to_string(),
            tasks_path: None,
        }]
    };

    Ok(CodingEvalConfig {
        tasksets,
        num_samples,
        timeout_secs,
        enable_pass2,
        enable_pass3,
        language_images,
        host_repo_path,
    })
}

fn parse_tasksets(value: &serde_yaml::Value) -> Result<Vec<TasksetConfig>> {
    let mut tasksets = Vec::new();
    let mapping = value
        .as_mapping()
        .ok_or_else(|| anyhow::anyhow!("coding_eval.tasksets must be a mapping"))?;
    for (key, val) in mapping {
        let name = key
            .as_str()
            .ok_or_else(|| anyhow::anyhow!("coding_eval taskset keys must be strings"))?
            .to_string();
        if val.is_null() {
            let task_type = TaskType::from_key(&name).ok_or_else(|| {
                anyhow::anyhow!(
                    "taskset '{}' uses '~' but its key is not a known type (human_eval, humaneval_plus, mbpp)",
                    name
                )
            })?;
            tasksets.push(TasksetConfig {
                name,
                task_type,
                language: "python".to_string(),
                tasks_path: None,
            });
            continue;
        }
        let map = val
            .as_mapping()
            .ok_or_else(|| anyhow::anyhow!("taskset '{}' must be '~' or a mapping", name))?;
        let type_name = get_yaml_str(map, "type")
            .map(ToOwned::to_owned)
            .or_else(|| TaskType::from_key(&name).map(|t| t.as_str().to_string()))
            .ok_or_else(|| anyhow::anyhow!("taskset '{}' is missing type", name))?;
        let task_type = TaskType::from_key(&type_name)
            .ok_or_else(|| anyhow::anyhow!("unknown coding_eval taskset type: {}", type_name))?;
        let language = get_yaml_str(map, "language")
            .unwrap_or("python")
            .to_string();
        let tasks_path = get_yaml_str(map, "tasks_path").map(PathBuf::from);
        tasksets.push(TasksetConfig {
            name,
            task_type,
            language,
            tasks_path,
        });
    }
    if tasksets.is_empty() {
        return Err(anyhow::anyhow!("coding_eval.tasksets cannot be empty"));
    }
    Ok(tasksets)
}

fn get_yaml_str<'a>(map: &'a serde_yaml::Mapping, key: &str) -> Option<&'a str> {
    map.get(serde_yaml::Value::String(key.to_string()))
        .and_then(|v| v.as_str())
}

fn load_tasks(taskset: &TasksetConfig) -> Result<Vec<CodingTask>> {
    let path = if let Some(path) = &taskset.tasks_path {
        path.clone()
    } else {
        download_taskset(taskset)?
    };
    let content = read_maybe_gz(&path)?;
    parse_jsonl_tasks(&content).with_context(|| format!("failed to parse {}", path.display()))
}

fn parse_jsonl_tasks(content: &str) -> Result<Vec<CodingTask>> {
    let mut tasks = Vec::new();
    for (idx, line) in content.lines().enumerate() {
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        let task: CodingTask = serde_json::from_str(line)
            .with_context(|| format!("invalid JSONL task at line {}", idx + 1))?;
        tasks.push(task);
    }
    Ok(tasks)
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
    if path.exists() {
        return Ok(path);
    }
    println!("  Downloading coding_eval taskset {}...", taskset.name);
    let bytes = reqwest::blocking::get(url)?.bytes()?;
    fs::write(&path, bytes)?;
    Ok(path)
}

fn read_maybe_gz(path: &Path) -> Result<String> {
    let bytes = fs::read(path)?;
    if path.extension() == Some(OsStr::new("gz")) {
        let mut decoder = GzDecoder::new(&bytes[..]);
        let mut content = String::new();
        decoder.read_to_string(&mut content)?;
        Ok(content)
    } else {
        Ok(String::from_utf8(bytes)?)
    }
}

fn build_initial_prompt(task: &CodingTask, language: &str) -> String {
    format!(
        "You are solving a {language} coding benchmark task.\n\nReturn ONLY {language} code. Do not include markdown fences, explanations, tests, or prose.\n\nImplement exactly this function name: `{}`. The solution must be self-contained and import any standard-library modules it needs.\n\nProblem prompt:\n{}\n",
        task.entry_point, task.prompt
    )
}

fn build_repair_prompt(
    task: &CodingTask,
    language: &str,
    previous_solution: &str,
    previous_result: Option<&DockerResult>,
) -> String {
    let summary = previous_result
        .map(|r| {
            format!(
                "Exit code: {:?}\nTimed out: {}\nError summary: {}\nSTDOUT:\n{}\nSTDERR:\n{}",
                r.exit_code,
                r.timed_out,
                r.error_summary,
                truncate(&r.stdout, 4000),
                truncate(&r.stderr, 4000)
            )
        })
        .unwrap_or_else(|| "No execution details available.".to_string());
    format!(
        "You are repairing a failed {language} coding benchmark solution.\n\nReturn ONLY {language} code. Do not include markdown fences, explanations, tests, or prose.\n\nYou must implement exactly this function name: `{}`.\n\nProblem prompt:\n{}\n\nPrevious solution:\n```{language}\n{}\n```\n\nThe previous solution failed with these execution details:\n{}\n\nProvide a corrected complete solution now.",
        task.entry_point, task.prompt, previous_solution, summary
    )
}

fn extract_code(response: &str) -> String {
    let python_re = Regex::new(r"(?is)```(?:python|py)\s*(.*?)\s*```").expect("valid regex");
    if let Some(caps) = python_re.captures(response) {
        return caps[1].trim().to_string();
    }
    let any_re = Regex::new(r"(?s)```[^\n`]*\s*(.*?)\s*```").expect("valid regex");
    if let Some(caps) = any_re.captures(response) {
        return caps[1].trim().to_string();
    }
    response.trim().to_string()
}

fn run_directory(model: &str, taskset: &str, task_id: &str, attempt: usize) -> PathBuf {
    Path::new("benchmark_results")
        .join("coding_eval_runs")
        .join(sanitize_path_component(model))
        .join(sanitize_path_component(taskset))
        .join(sanitize_path_component(task_id))
        .join(format!("attempt_{}", attempt))
}

fn sanitize_path_component(value: &str) -> String {
    value
        .chars()
        .map(|c| {
            if c.is_ascii_alphanumeric() || c == '-' || c == '_' || c == '.' {
                c
            } else {
                '_'
            }
        })
        .collect()
}

fn write_harness(
    run_dir: &Path,
    task: &CodingTask,
    task_type: &TaskType,
    solution: &str,
) -> Result<HarnessFiles> {
    fs::create_dir_all(run_dir)?;
    let solution_path = run_dir.join("solution.py");
    fs::write(&solution_path, solution)?;
    let test_path = run_dir.join("test_solution.py");
    let test_content = match task_type {
        TaskType::HumanEval => build_human_eval_harness(task)?,
        TaskType::HumanEvalPlus | TaskType::Mbpp => build_evalplus_harness(task, task_type)?,
    };
    fs::write(&test_path, test_content)?;
    Ok(HarnessFiles { test_path })
}

fn build_human_eval_harness(task: &CodingTask) -> Result<String> {
    let test = task
        .test
        .as_ref()
        .ok_or_else(|| anyhow::anyhow!("HumanEval task {} missing test", task.task_id))?;
    Ok(format!(
        r#"import traceback
from solution import {entry} as candidate

{test}

if __name__ == "__main__":
    try:
        check(candidate)
        print("PASS")
    except BaseException:
        traceback.print_exc()
        raise
"#,
        entry = task.entry_point,
        test = test
    ))
}

fn build_evalplus_harness(task: &CodingTask, task_type: &TaskType) -> Result<String> {
    let canonical = task.canonical_solution.as_ref().ok_or_else(|| {
        anyhow::anyhow!("EvalPlus task {} missing canonical_solution", task.task_id)
    })?;
    let reference_source = if canonical.contains(&format!("def {}", task.entry_point)) {
        canonical.clone()
    } else {
        format!("{}\n{}", task.prompt, canonical)
    };
    let base_input = serde_json::to_string(task.base_input.as_ref().unwrap_or(&Vec::new()))?;
    let plus_input = serde_json::to_string(task.plus_input.as_ref().unwrap_or(&Vec::new()))?;
    let assertion = task.assertion.clone().unwrap_or_default();
    let run_assertion = matches!(task_type, TaskType::Mbpp) && !assertion.trim().is_empty();
    let assertion_literal = serde_json::to_string(&assertion)?;
    Ok(format!(
        r#"import copy
import json
import math
import traceback
import types
from solution import {entry} as candidate

REFERENCE_SOURCE = {reference_source:?}
BASE_INPUTS = json.loads({base_input:?})
PLUS_INPUTS = json.loads({plus_input:?})
ATOL = {atol}
ASSERTION = json.loads({assertion_literal:?})
RUN_ASSERTION = {run_assertion}
ENTRY_POINT = {entry:?}

reference_module = types.ModuleType("reference_solution")
exec(REFERENCE_SOURCE, reference_module.__dict__)
reference = getattr(reference_module, ENTRY_POINT)

def equivalent(actual, expected, atol=0.0):
    if isinstance(actual, float) or isinstance(expected, float):
        try:
            return math.isclose(actual, expected, rel_tol=1e-09, abs_tol=atol)
        except TypeError:
            return False
    if isinstance(actual, (list, tuple)) and isinstance(expected, (list, tuple)):
        return len(actual) == len(expected) and all(equivalent(a, e, atol) for a, e in zip(actual, expected))
    if isinstance(actual, dict) and isinstance(expected, dict):
        return actual.keys() == expected.keys() and all(equivalent(actual[k], expected[k], atol) for k in actual)
    if isinstance(actual, set) and isinstance(expected, set):
        return actual == expected
    return actual == expected

def run_case(args):
    if not isinstance(args, list):
        args = [args]
    cand_args = copy.deepcopy(args)
    ref_args = copy.deepcopy(args)
    actual = candidate(*cand_args)
    expected = reference(*ref_args)
    assert equivalent(actual, expected, ATOL), f"for args={{args!r}} expected {{expected!r}} got {{actual!r}}"

def main():
    if RUN_ASSERTION:
        globals()[ENTRY_POINT] = candidate
        exec(ASSERTION, globals())
    for args in BASE_INPUTS:
        run_case(args)
    for args in PLUS_INPUTS:
        run_case(args)
    print("PASS")

if __name__ == "__main__":
    try:
        main()
    except BaseException:
        traceback.print_exc()
        raise
"#,
        entry = task.entry_point,
        reference_source = reference_source,
        base_input = base_input,
        plus_input = plus_input,
        atol = task.atol.unwrap_or(0.0),
        assertion_literal = assertion_literal,
        run_assertion = if run_assertion { "True" } else { "False" },
    ))
}

fn run_in_docker(
    test_path: &Path,
    run_dir: &Path,
    docker_image: &str,
    timeout_secs: u64,
    host_repo_path: Option<&Path>,
) -> Result<DockerResult> {
    let host_task_dir = docker_mount_source(run_dir, host_repo_path)?;
    let container_name = docker_container_name();
    let test_file = test_path
        .file_name()
        .and_then(|s| s.to_str())
        .unwrap_or("test_solution.py");
    let mut child = Command::new("docker")
        .arg("run")
        .arg("--rm")
        .arg("--name")
        .arg(&container_name)
        .arg("--network")
        .arg("none")
        .arg("--read-only")
        .arg("--tmpfs")
        .arg("/tmp:rw,noexec,nosuid,size=64m")
        .arg("--cap-drop=ALL")
        .arg("--security-opt=no-new-privileges")
        .arg("--pids-limit")
        .arg("128")
        .arg("--memory")
        .arg("512m")
        .arg("-v")
        .arg(format!("{}:/work:ro", host_task_dir.display()))
        .arg("-w")
        .arg("/work")
        .arg("-e")
        .arg("PYTHONDONTWRITEBYTECODE=1")
        .arg(docker_image)
        .arg("python")
        .arg(test_file)
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .with_context(|| "failed to start docker; is Docker installed and available on PATH?")?;

    let deadline = Instant::now() + Duration::from_secs(timeout_secs);
    loop {
        if child.try_wait()?.is_some() {
            let output = child.wait_with_output()?;
            let stdout = String::from_utf8_lossy(&output.stdout).to_string();
            let stderr = String::from_utf8_lossy(&output.stderr).to_string();
            let passed = output.status.success();
            let exit_code = output.status.code();
            let error_summary = if passed {
                "passed".to_string()
            } else {
                docker_error_summary(&stderr, host_repo_path.is_some())
            };
            return Ok(DockerResult {
                passed,
                timed_out: false,
                exit_code,
                stdout,
                stderr,
                error_summary,
            });
        }
        if Instant::now() >= deadline {
            let _ = Command::new("docker")
                .arg("rm")
                .arg("-f")
                .arg(&container_name)
                .output();
            let _ = child.kill();
            let output = child.wait_with_output()?;
            return Ok(DockerResult {
                passed: false,
                timed_out: true,
                exit_code: None,
                stdout: String::from_utf8_lossy(&output.stdout).to_string(),
                stderr: String::from_utf8_lossy(&output.stderr).to_string(),
                error_summary: format!("timed out after {} seconds", timeout_secs),
            });
        }
        std::thread::sleep(Duration::from_millis(100));
    }
}

fn docker_container_name() -> String {
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_nanos();
    format!(
        "llm-benchmark-runner-coding-eval-{}-{}",
        std::process::id(),
        nanos
    )
}

fn docker_mount_source(run_dir: &Path, host_repo_path: Option<&Path>) -> Result<PathBuf> {
    let canonical_run_dir = run_dir.canonicalize()?;
    if let Some(host_repo_path) = host_repo_path {
        let repo_root = std::env::current_dir()?.canonicalize()?;
        let relative = canonical_run_dir
            .strip_prefix(&repo_root)
            .with_context(|| {
                format!(
                    "run dir {} is not under repo root {}; cannot apply host_repo_path",
                    canonical_run_dir.display(),
                    repo_root.display()
                )
            })?;
        Ok(host_repo_path.join(relative))
    } else {
        Ok(canonical_run_dir)
    }
}

fn docker_error_summary(stderr: &str, host_repo_path_set: bool) -> String {
    let mut summary = truncate(stderr.trim(), 1200);
    if !host_repo_path_set
        && (stderr.contains("Mounts denied")
            || stderr.contains("bind source path does not exist")
            || stderr.contains("invalid mount config")
            || stderr.contains("no such file or directory"))
    {
        summary.push_str("\nHint: if llm-benchmark-runner is running inside Docker with the host Docker socket mounted, set benchmark.coding_eval.host_repo_path to the host-visible repository path.");
    }
    if summary.is_empty() {
        "docker run failed".to_string()
    } else {
        summary
    }
}

fn attempts_to_json(attempts: &[AttemptOutcome]) -> Vec<serde_json::Value> {
    attempts
        .iter()
        .map(|attempt| {
            serde_json::json!({
                "attempt": attempt.attempt,
                "skipped": attempt.skipped,
                "passed": attempt.passed,
                "timed_out": attempt.timed_out,
                "exit_code": attempt.exit_code,
                "stdout": truncate(&attempt.stdout, 4000),
                "stderr": truncate(&attempt.stderr, 4000),
                "error_summary": attempt.error_summary,
                "output_tokens": attempt.output_tokens,
                "thinking_tokens": attempt.thinking_tokens,
            })
        })
        .collect()
}

fn taskset_stats_to_json(
    stats: &HashMap<String, TasksetStats>,
    max_attempts: usize,
) -> HashMap<String, serde_json::Value> {
    stats
        .iter()
        .map(|(name, stat)| {
            let mut map = serde_json::Map::new();
            map.insert("total".to_string(), serde_json::json!(stat.total));
            map.insert(
                "passed".to_string(),
                serde_json::json!(stat.pass_counts[max_attempts - 1]),
            );
            map.insert(
                "pass_at_1".to_string(),
                serde_json::json!(rate(stat.pass_counts[0], stat.total)),
            );
            if max_attempts >= 2 {
                map.insert(
                    "pass_at_2".to_string(),
                    serde_json::json!(rate(stat.pass_counts[1], stat.total)),
                );
            }
            if max_attempts >= 3 {
                map.insert(
                    "pass_at_3".to_string(),
                    serde_json::json!(rate(stat.pass_counts[2], stat.total)),
                );
            }
            map.insert(
                "timeout_count".to_string(),
                serde_json::json!(stat.timeout_count),
            );
            map.insert(
                "skipped_later_attempts".to_string(),
                serde_json::json!(stat.skipped_later_attempts),
            );
            (name.clone(), serde_json::Value::Object(map))
        })
        .collect()
}

fn rate(passed: usize, total: usize) -> f64 {
    if total == 0 {
        0.0
    } else {
        passed as f64 / total as f64
    }
}

fn truncate(value: &str, max_chars: usize) -> String {
    if value.chars().count() <= max_chars {
        return value.to_string();
    }
    let mut out: String = value.chars().take(max_chars).collect();
    out.push('…');
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn extracts_python_fence_first() {
        let response = "text\n```python\ndef foo():\n    return 1\n```\nmore";
        assert_eq!(extract_code(response), "def foo():\n    return 1");
    }

    #[test]
    fn extracts_any_fence_then_plain_text() {
        assert_eq!(extract_code("```\ndef foo(): pass\n```"), "def foo(): pass");
        assert_eq!(extract_code("def foo():\n    pass"), "def foo():\n    pass");
    }

    #[test]
    fn parses_humaneval_jsonl() {
        let content = r#"{"task_id":"HumanEval/0","prompt":"def f():\n    pass","entry_point":"f","test":"def check(candidate):\n    assert candidate() == 1"}"#;
        let tasks = parse_jsonl_tasks(content).unwrap();
        assert_eq!(tasks.len(), 1);
        assert_eq!(tasks[0].entry_point, "f");
    }

    #[test]
    fn parses_evalplus_jsonl() {
        let content = r#"{"task_id":"HumanEval/0","prompt":"def f(x):\n    ","entry_point":"f","canonical_solution":"return x","base_input":[[1]],"plus_input":[[2]],"atol":0}"#;
        let tasks = parse_jsonl_tasks(content).unwrap();
        assert_eq!(tasks[0].base_input.as_ref().unwrap().len(), 1);
        assert_eq!(tasks[0].plus_input.as_ref().unwrap().len(), 1);
    }

    #[test]
    fn parses_taskset_shorthand_and_custom() {
        let yaml: serde_yaml::Value = serde_yaml::from_str(
            r#"
tasksets:
  humaneval_plus: ~
  custom:
    type: human_eval
    language: python
    tasks_path: ./custom.jsonl
"#,
        )
        .unwrap();
        let cfg = parse_config(&yaml).unwrap();
        assert_eq!(cfg.tasksets.len(), 2);
        assert_eq!(cfg.tasksets[0].task_type, TaskType::HumanEvalPlus);
        assert_eq!(
            cfg.tasksets[1].tasks_path.as_deref(),
            Some(Path::new("./custom.jsonl"))
        );
    }

    #[test]
    fn human_eval_harness_contains_check_call() {
        let task = CodingTask {
            task_id: "t".to_string(),
            prompt: "def f(): pass".to_string(),
            entry_point: "f".to_string(),
            test: Some("def check(candidate):\n    assert candidate() == 1".to_string()),
            canonical_solution: None,
            base_input: None,
            plus_input: None,
            atol: None,
            assertion: None,
        };
        let harness = build_human_eval_harness(&task).unwrap();
        assert!(harness.contains("check(candidate)"));
        assert!(harness.contains("from solution import f as candidate"));
    }

    #[test]
    fn evalplus_harness_compares_candidate_to_reference() {
        let task = CodingTask {
            task_id: "t".to_string(),
            prompt: "def f(x):\n    ".to_string(),
            entry_point: "f".to_string(),
            test: None,
            canonical_solution: Some("return x + 1".to_string()),
            base_input: Some(vec![serde_json::json!([1])]),
            plus_input: Some(vec![serde_json::json!([2])]),
            atol: Some(0.0),
            assertion: None,
        };
        let harness = build_evalplus_harness(&task, &TaskType::HumanEvalPlus).unwrap();
        assert!(harness.contains("reference = getattr(reference_module, ENTRY_POINT)"));
        assert!(harness.contains("run_case(args)"));
        assert!(harness.contains("PLUS_INPUTS"));
    }

    #[test]
    fn docker_error_summary_mentions_host_repo_path_for_mount_errors() {
        let summary = docker_error_summary("bind source path does not exist", false);
        assert!(summary.contains("host_repo_path"));
    }
}
