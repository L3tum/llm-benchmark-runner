use crate::benchmarks::Benchmark;
use crate::config::Model;
use crate::docker_runner::{DockerMount, DockerRunConfig, DockerRunner};
use crate::reports::report_helpers::build_accuracy_report;
use crate::shared::{BenchmarkCategory, BenchmarkResult, TaskResult};
use crate::token_tracker::TokenTracker;
use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use serde_json::Value as JsonValue;
use std::collections::HashMap;
use std::fs;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::Mutex;

/// MultiPL-E benchmark: HumanEval translated to multiple programming languages.
/// Downloads prompts from HuggingFace (nuprl/MultiPL-E) and evaluates via Docker.
pub struct MultiPLEBenchmark {
    state: Mutex<MultiPLEState>,
}

struct MultiPLEState {
    items: Vec<MultiPLEInstance>,
    current_idx: usize,
    config: Option<MultiPLEConfig>,
    /// Generated code per task, populated during execute_one for batch evaluation
    generated_code: HashMap<String, String>,
}

impl Default for MultiPLEBenchmark {
    fn default() -> Self {
        Self {
            state: Mutex::new(MultiPLEState {
                items: Vec::new(),
                current_idx: 0,
                config: None,
                generated_code: HashMap::new(),
            }),
        }
    }
}

#[derive(Debug, Clone, Deserialize, Serialize)]
struct MultiPLEInstance {
    #[serde(rename = "task_id")]
    task_id: String,
    prompt: String,
    canonical_solution: String,
    test: String,
    code_v: String,
    #[serde(default)]
    language: String,
}

#[derive(Debug, Clone)]
struct MultiPLEConfig {
    languages: Vec<String>,
    num_samples: Option<usize>,
    timeout_secs: u64,
    eval_image: String,
}

const MULTIPLE_DATASET: &str = "nuprl/MultiPL-E";
const MULTIPLE_DEFAULT_IMAGE: &str = "ghcr.io/nuprl/multipl-e-evaluation";

/// Language codes used by MultiPL-E for their dataset files.
/// These map to the suffix in the HuggingFace dataset config name (e.g., "humaneval-cs-reworded").
const LANGUAGE_CODES: &[(&str, &str)] = &[
    ("ada", "Ada"),
    ("clj", "Clojure"),
    ("coq", "Coq"),
    ("cpp", "C++"),
    ("cs", "C#"),
    ("d", "D"),
    ("dafny", "Dafny"),
    ("dart", "Dart"),
    ("elixir", "Elixir"),
    ("fs", "F#"),
    ("go", "Go"),
    ("hs", "Haskell"),
    ("java", "Java"),
    ("jl", "Julia"),
    ("js", "JavaScript"),
    ("lean", "Lean"),
    ("lua", "Lua"),
    ("luau", "Luau"),
    ("ml", "OCaml"),
    ("php", "PHP"),
    ("pl", "Perl"),
    ("py", "Python"),
    ("r", "R"),
    ("rb", "Ruby"),
    ("rkt", "Racket"),
    ("rs", "Rust"),
    ("scala", "Scala"),
    ("sh", "Shell"),
    ("swift", "Swift"),
    ("ts", "TypeScript"),
    ("matlab", "MATLAB"),
];

impl Benchmark for MultiPLEBenchmark {
    fn name(&self) -> &str {
        "multipl_e"
    }

    fn display_name(&self) -> &'static str {
        "MultiPL-E (HumanEval Multilingual)"
    }

    fn category(&self) -> BenchmarkCategory {
        BenchmarkCategory::ShortContextCoding
    }

    fn pre_execute(&self, config: &yaml_serde::Value) -> Result<()> {
        let cfg = parse_config(config)?;
        let items = download_and_filter(&cfg)?;
        println!(
            "MultiPL-E: {} instances across {} languages",
            items.len(),
            cfg.languages.len()
        );

        let mut state = self.state.lock().expect(crate::shared::MUTEX_PANIC_MSG);
        state.items = items;
        state.current_idx = 0;
        state.config = Some(cfg);
        drop(state);

        Ok(())
    }

    fn execute_one(
        &self,
        model: &Model,
        _config: &yaml_serde::Value,
        tracker: &mut TokenTracker,
    ) -> Result<Option<TaskResult>> {
        let (instance, task_id, _cfg) = {
            let mut state = self.state.lock().expect(crate::shared::MUTEX_PANIC_MSG);
            if state.current_idx >= state.items.len() {
                return Ok(None);
            }
            let idx = state.current_idx;
            state.current_idx += 1;
            let item = state.items[idx].clone();
            let cfg = state.config.clone().ok_or_else(|| {
                anyhow::anyhow!("MultiPL-E config not initialized, did pre_execute run?")
            })?;
            (item, format!("task-{}", idx), cfg)
        };

        // Build the prompt — send the translated HumanEval prompt to the model
        let system_prompt = "You are a coding assistant. Complete the function body based on the docstring and function signature. \
            Return ONLY the completed function code. Do not include any explanation or markdown fences.";

        let response =
            tracker.chat_completion(&model.model_name, system_prompt, &instance.prompt)?;

        // Extract code from the response
        let generated_code = extract_code(&response);

        // Store generated code for batch evaluation
        {
            let mut state = self.state.lock().expect(crate::shared::MUTEX_PANIC_MSG);
            state
                .generated_code
                .insert(task_id.clone(), generated_code.clone());
        }

        // Return placeholder result; batch_evaluate will re-evaluate all tasks grouped by language
        Ok(Some(
            TaskResult::new(task_id, false, 0.0, vec![instance.language.clone()]).with_metadata(
                Some(serde_json::json!({
                    "task_id": instance.task_id,
                    "language": instance.language,
                    "prompt": instance.prompt,
                })),
            ),
        ))
    }

    fn to_report_result(&self, b: &BenchmarkResult) -> Result<BenchmarkResult> {
        build_accuracy_report(
            b,
            "pass@1",
            "By Language",
            "Pass@1 by Programming Language",
            Some("passed"),
        )
    }

    fn batch_evaluate(
        &self,
        task_results: &[TaskResult],
        _config: &yaml_serde::Value,
    ) -> Result<Option<Vec<TaskResult>>> {
        let state = self.state.lock().expect(crate::shared::MUTEX_PANIC_MSG);

        // If no generated code was stored, nothing to batch-evaluate
        if state.generated_code.is_empty() {
            return Ok(None);
        }

        let cfg = state.config.clone().ok_or_else(|| {
            anyhow::anyhow!("MultiPL-E config not available for batch evaluation")
        })?;

        // Build O(1) lookup: task_id → instance
        let items_by_id: HashMap<&str, &MultiPLEInstance> = state
            .items
            .iter()
            .map(|item| (item.task_id.as_str(), item))
            .collect();

        // Group task results by language, pairing with generated code
        let mut lang_batches: HashMap<String, Vec<(MultiPLEInstance, String, String)>> =
            HashMap::new();

        for tr in task_results.iter() {
            if let Some(code) = state.generated_code.get(&tr.task_id) {
                if let Some(item) = items_by_id.get(tr.task_id.as_str()) {
                    let lang = item.language.clone();
                    lang_batches.entry(lang).or_default().push((
                        (*item).clone(),
                        code.clone(),
                        tr.task_id.clone(),
                    ));
                }
            }
        }

        drop(state); // Release lock before Docker calls

        // Evaluate each language batch in a single Docker container
        let mut result_map: HashMap<String, bool> = HashMap::new();

        for (_lang, batch_items) in lang_batches {
            let instance_code_pairs: Vec<(MultiPLEInstance, String)> = batch_items
                .iter()
                .cloned()
                .map(|(inst, code, _)| (inst, code))
                .collect();

            let results = evaluate_batch_in_docker(&instance_code_pairs, &cfg)?;

            for ((_, _, task_id), passed) in batch_items.iter().zip(results.iter()) {
                result_map.insert(task_id.clone(), *passed);
            }
        }

        // Merge results: update passed/score, preserving token counts from runner
        let updated: Vec<TaskResult> = task_results
            .iter()
            .map(|tr| {
                if let Some(&passed) = result_map.get(&tr.task_id) {
                    let mut updated = tr.clone();
                    updated.passed = passed;
                    updated.score = if passed { 1.0 } else { 0.0 };
                    updated
                } else {
                    tr.clone()
                }
            })
            .collect();

        Ok(Some(updated))
    }
}

fn parse_config(config: &yaml_serde::Value) -> Result<MultiPLEConfig> {
    let docker_cfg = config.get("__docker");

    let languages = config
        .get("languages")
        .and_then(|v| v.as_sequence())
        .map(|seq| {
            seq.iter()
                .filter_map(|v| v.as_str().map(|s| s.to_lowercase()))
                .collect::<Vec<String>>()
        })
        .unwrap_or_else(|| {
            // Default to all available languages
            LANGUAGE_CODES
                .iter()
                .map(|(code, _)| (*code).to_string())
                .collect()
        });

    let num_samples = config
        .get("num_samples")
        .and_then(|v| v.as_i64())
        .filter(|&n| n > 0 && n <= 10_000) // 164 tasks * 31 langs ≈ 5084; 10K gives headroom
        .map(|n| n as usize);

    let timeout_secs = config
        .get("timeout_secs")
        .and_then(|v| v.as_i64())
        .unwrap_or(30)
        .clamp(1, 600) as u64; // cap at 10 minutes

    let eval_image = docker_cfg
        .and_then(|docker| docker.get("images"))
        .and_then(|images| images.get("multipl_e_eval"))
        .and_then(|v| v.as_str())
        .unwrap_or(MULTIPLE_DEFAULT_IMAGE)
        .to_string();

    Ok(MultiPLEConfig {
        languages,
        num_samples,
        timeout_secs,
        eval_image,
    })
}

fn download_and_filter(cfg: &MultiPLEConfig) -> Result<Vec<MultiPLEInstance>> {
    use std::thread;

    const MAX_CONCURRENT_DOWNLOADS: usize = 8;
    let languages: Vec<String> = cfg.languages.clone();
    let num_samples = cfg.num_samples;

    // Download datasets in parallel using chunked batched thread scopes (rate limiting)
    let mut all_items: Vec<MultiPLEInstance> = Vec::new();

    for chunk in languages.chunks(MAX_CONCURRENT_DOWNLOADS) {
        thread::scope(|s| {
            let mut handles = Vec::new();
            for lang in chunk {
                println!(
                    "  MultiPL-E: downloading humaneval-{}-reworded for language '{}'",
                    lang, lang
                );
                let config_name = format!("humaneval-{}-reworded", lang);
                handles.push(s.spawn(move || -> Option<Vec<MultiPLEInstance>> {
                    match download_language_dataset(lang, &config_name) {
                        Ok(mut items) => {
                            if let Some(n) = num_samples {
                                if n < items.len() {
                                    items.truncate(n);
                                }
                            }
                            for item in &mut items {
                                item.language = lang.clone();
                            }
                            Some(items)
                        }
                        Err(e) => {
                            eprintln!(
                                "  MultiPL-E: failed to download dataset for '{}': {}",
                                lang, e
                            );
                            None
                        }
                    }
                }));
            }
            for handle in handles {
                if let Some(items) = handle.join().unwrap() {
                    all_items.extend(items);
                }
            }
        });
    }

    Ok(all_items)
}

fn download_language_dataset(lang: &str, config_name: &str) -> Result<Vec<MultiPLEInstance>> {
    let cache_path = dataset_cache_path(lang);

    // Check cache first
    if cache_path.exists() {
        return read_cached_dataset(&cache_path);
    }

    // Download from HuggingFace datasets server
    println!(
        "  Downloading MultiPL-E dataset for {} from HuggingFace...",
        lang
    );
    let client = reqwest::blocking::Client::builder().build()?;

    let url = reqwest::Url::parse_with_params(
        "https://datasets-server.huggingface.co/rows",
        &[
            ("dataset", MULTIPLE_DATASET),
            ("config", config_name),
            ("split", "test"),
            ("offset", "0"),
            ("length", "164"), // HumanEval has exactly 164 tasks
        ],
    )?;

    let response: JsonValue = client.get(url).send()?.error_for_status()?.json()?;

    let rows = response
        .get("rows")
        .and_then(|v| v.as_array())
        .cloned()
        .unwrap_or_default();

    let mut instances = Vec::new();
    for row_value in &rows {
        let row = row_value
            .get("row")
            .cloned()
            .ok_or_else(|| anyhow::anyhow!("missing row in MultiPL-E dataset response"))?;
        instances.push(serde_json::from_value::<MultiPLEInstance>(row)?);
    }

    // Validate downloaded field sizes to prevent resource exhaustion
    const MAX_FIELD_SIZE: usize = 128 * 1024;
    for instance in &instances {
        if instance.prompt.len() > MAX_FIELD_SIZE {
            return Err(anyhow::anyhow!(
                "Instance prompt exceeds max size ({} > {} bytes)",
                instance.prompt.len(),
                MAX_FIELD_SIZE
            ));
        }
        if instance.test.len() > MAX_FIELD_SIZE {
            return Err(anyhow::anyhow!(
                "Instance test exceeds max size ({} > {} bytes)",
                instance.test.len(),
                MAX_FIELD_SIZE
            ));
        }
        if instance.canonical_solution.len() > MAX_FIELD_SIZE {
            return Err(anyhow::anyhow!(
                "Instance canonical_solution exceeds max size ({} > {} bytes)",
                instance.canonical_solution.len(),
                MAX_FIELD_SIZE
            ));
        }
        if instance.code_v.len() > MAX_FIELD_SIZE {
            return Err(anyhow::anyhow!(
                "Instance code_v exceeds max size ({} > {} bytes)",
                instance.code_v.len(),
                MAX_FIELD_SIZE
            ));
        }
    }

    // Cache the downloaded data
    if let Some(parent) = cache_path.parent() {
        fs::create_dir_all(parent)?;
    }
    let tmp_path = cache_path.with_extension(format!("jsonl.tmp.{}", std::process::id()));
    let mut file = fs::File::create(&tmp_path)?;
    for instance in &instances {
        writeln!(file, "{}", serde_json::to_string(instance)?)?;
    }
    fs::rename(&tmp_path, &cache_path).inspect_err(|_err| {
        let _ = fs::remove_file(&tmp_path);
    })?;

    Ok(instances)
}

fn dataset_cache_path(lang: &str) -> PathBuf {
    dirs::cache_dir()
        .unwrap_or_default()
        .join("llm-benchmark-runner")
        .join("multipl_e")
        .join(format!("humaneval-{}.jsonl", lang))
}

fn read_cached_dataset(path: &Path) -> Result<Vec<MultiPLEInstance>> {
    let content = fs::read_to_string(path)?;
    content
        .lines()
        .filter(|line| !line.trim().is_empty())
        .map(|line| {
            serde_json::from_str(line)
                .with_context(|| format!("invalid MultiPL-E cache row in {}", path.display()))
        })
        .collect()
}

/// Extract code from the model response, stripping markdown fences and explanations.
fn extract_code(response: &str) -> String {
    let trimmed = response.trim();

    // Try to extract from markdown code fences first
    if let Some(first_fence) = trimmed.find("```") {
        let after_first = &trimmed[first_fence + 3..];

        // Check if there's a language identifier on the first line (e.g., ```python)
        let (content_start, rest) = if let Some(newline_pos) = after_first.find('\n') {
            let first_line = after_first[..newline_pos].trim();
            if first_line.is_empty()
                || first_line
                    .chars()
                    .all(|c| c.is_ascii_alphabetic() || c == '#' || c == '-' || c.is_ascii_digit())
            {
                // Language identifier line
                (
                    &after_first[newline_pos + 1..],
                    &after_first[newline_pos + 1..],
                )
            } else {
                // Content starts immediately after ```
                (after_first, after_first)
            }
        } else {
            // No newline, all content is on one line after ```
            (after_first, after_first)
        };

        if let Some(second_fence) = rest.find("```") {
            return content_start[..second_fence].trim().to_string();
        }
        // Unclosed fence: return content_start (already stripped opening fence + lang id)
        return content_start.trim().to_string();
    }

    trimmed.to_string()
}

/// Evaluate a single instance via Docker.
/// Kept for debugging single-instance evaluation; batch evaluation uses `evaluate_batch_in_docker()`.
#[allow(dead_code)]
fn evaluate_in_docker(
    instance: &MultiPLEInstance,
    generated_code: &str,
    cfg: &MultiPLEConfig,
) -> Result<bool> {
    // Validate generated code size to prevent disk exhaustion in the container
    const MAX_CODE_SIZE: usize = 64 * 1024; // 64KB
    if generated_code.len() > MAX_CODE_SIZE {
        return Err(anyhow::anyhow!(
            "Generated code exceeds maximum size ({}/{} bytes)",
            generated_code.len(),
            MAX_CODE_SIZE
        ));
    }

    // Create a temporary directory that auto-cleans when dropped
    let run_dir = tempfile::tempdir_in(
        dirs::cache_dir()
            .unwrap_or_default()
            .join("llm-benchmark-runner")
            .join("multipl_e_runs"),
    )
    .context("failed to create temp dir for MultiPL-E evaluation")?;

    // Build the complete program: prompt + generated code + tests
    // For Python, replace the `pass` keyword with the generated code.
    // The MultiPL-E dataset fields already contain language-specific wrappers:
    // - prompt: imports + class/method signature + opening braces
    // - test: test assertions + closing braces
    // - generated_code: the model's function body (goes between them)
    let program = if instance.language == "py" {
        format!(
            "{}{}",
            replace_python_pass(&instance.prompt, generated_code),
            instance.test
        )
    } else {
        // For other languages, concatenate prompt + code + tests.
        // The container's language-specific evaluator handles compilation
        // and execution (javac, g++, rustc, etc.).
        format!("{}{}{}", instance.prompt, generated_code, instance.test)
    };

    // Write the program to a file for the container evaluator
    fs::write(run_dir.path().join("program"), &program)?;

    // Write a Python evaluation wrapper that calls the container's built-in
    // language-specific evaluators from containerized_eval.py
    let eval_script = r#"
import json
import sys
from containerized_eval import eval_string_script

program = open("/work/program").read()
language = sys.argv[1]

try:
    result = eval_string_script(language, program)
except Exception as e:
    print(json.dumps({"passed": False, "error": str(e)}))
    sys.exit(1)

print(json.dumps({"passed": result["status"] == "OK"}))
"#;
    fs::write(run_dir.path().join("eval.py"), eval_script)?;

    // Run evaluation in Docker using the container's Python evaluator
    let mut docker = DockerRunConfig::new(
        &cfg.eval_image,
        vec![
            "python3".to_string(),
            "/work/eval.py".to_string(),
            instance.language.clone(),
        ],
        cfg.timeout_secs,
    );
    docker
        .mounts
        .push(DockerMount::readwrite(run_dir.path(), "/work"));
    docker.workdir = Some("/work".to_string());
    docker.network_none = true;
    docker.read_only_root = true;
    docker.name_prefix = "llm-benchmark-runner-multipl-e".to_string();

    let output = DockerRunner::run(&docker)
        .with_context(|| format!("Docker evaluation failed for task {}", instance.task_id))?;

    if output.success() {
        // Parse JSON result: {"passed": true/false, ...}
        let stdout = output.stdout.trim();
        if let Ok(result) = serde_json::from_str::<serde_json::Value>(stdout) {
            Ok(result
                .get("passed")
                .and_then(|v| v.as_bool())
                .unwrap_or(false))
        } else {
            // If we can't parse, the evaluation likely failed
            println!(
                "  MultiPL-E: unexpected eval output for {}: {}",
                instance.task_id, stdout
            );
            Ok(false)
        }
    } else {
        // Container exited with error or timed out
        Ok(false)
    }
}

/// Evaluate multiple instances of the same language in a single Docker container call.
/// This dramatically reduces container launch overhead compared to evaluating one at a time.
///
/// # Arguments
/// * `instances` - Vector of (instance, generated_code) pairs, all for the same language
/// * `cfg` - The MultiPL-E config with Docker settings
///
/// # Returns
/// A vector of booleans indicating whether each instance passed, in the same order as input.
fn evaluate_batch_in_docker(
    batch: &[(MultiPLEInstance, String)],
    cfg: &MultiPLEConfig,
) -> Result<Vec<bool>> {
    // All instances in the batch must be the same language
    let language = batch
        .first()
        .map(|(inst, _)| inst.language.as_str())
        .ok_or_else(|| anyhow::anyhow!("Empty batch"))?;

    // Validate generated code sizes
    const MAX_CODE_SIZE: usize = 64 * 1024;
    for (inst, code) in batch {
        if code.len() > MAX_CODE_SIZE {
            return Err(anyhow::anyhow!(
                "Generated code exceeds maximum size ({}/{} bytes) for task {}",
                code.len(),
                MAX_CODE_SIZE,
                inst.task_id
            ));
        }
    }

    // Create a temporary directory for this batch
    let run_dir = tempfile::tempdir_in(
        dirs::cache_dir()
            .unwrap_or_default()
            .join("llm-benchmark-runner")
            .join("multipl_e_runs"),
    )
    .context("failed to create temp dir for MultiPL-E batch evaluation")?;

    // Write each program to a separate file
    for (i, (instance, generated_code)) in batch.iter().enumerate() {
        let program = if instance.language == "py" {
            format!(
                "{}{}",
                replace_python_pass(&instance.prompt, generated_code),
                instance.test
            )
        } else {
            format!("{}{}{}", instance.prompt, generated_code, instance.test)
        };
        fs::write(run_dir.path().join(format!("program_{i}")), &program)?;
    }

    // Write a batch evaluation script that processes all programs
    let batch_script = r#"
import json
import sys
from containerized_eval import eval_string_script

language = sys.argv[1]
num_tasks = int(sys.argv[2])

results = []
for i in range(num_tasks):
    program = open(f"/work/program_{i}").read()
    try:
        result = eval_string_script(language, program)
        passed = result["status"] == "OK"
    except Exception as e:
        passed = False
    results.append({"index": i, "passed": passed})

print(json.dumps(results))
"#;
    fs::write(run_dir.path().join("batch_eval.py"), batch_script)?;

    // Run batch evaluation in Docker
    let mut docker = DockerRunConfig::new(
        &cfg.eval_image,
        vec![
            "python3".to_string(),
            "/work/batch_eval.py".to_string(),
            language.to_string(),
            batch.len().to_string(),
        ],
        // Scale timeout with batch size, capped at 4 hours to prevent runaway execution
        cfg.timeout_secs
            .max(1)
            .saturating_mul(batch.len() as u64)
            .min(14400),
    );
    docker
        .mounts
        .push(DockerMount::readwrite(run_dir.path(), "/work"));
    docker.workdir = Some("/work".to_string());
    docker.network_none = true;
    docker.read_only_root = true;
    docker.name_prefix = "llm-benchmark-runner-multipl-e-batch".to_string();

    let output = DockerRunner::run(&docker)
        .with_context(|| format!("Docker batch evaluation failed for {} tasks", batch.len()))?;

    if output.success() {
        let stdout = output.stdout.trim();
        if let Ok(results) = serde_json::from_str::<Vec<serde_json::Value>>(stdout) {
            let mut passed_map = std::collections::HashMap::new();
            for r in &results {
                if let (Some(idx), Some(passed)) = (
                    r.get("index").and_then(|v| v.as_u64()),
                    r.get("passed").and_then(|v| v.as_bool()),
                ) {
                    passed_map.insert(idx, passed);
                }
            }
            Ok(batch
                .iter()
                .enumerate()
                .map(|(i, _)| passed_map.get(&(i as u64)).copied().unwrap_or(false))
                .collect())
        } else {
            println!(
                "MultiPL-E: unexpected batch eval output: {}",
                stdout.chars().take(200).collect::<String>()
            );
            Ok(vec![false; batch.len()])
        }
    } else {
        // Container error or timeout
        Ok(vec![false; batch.len()])
    }
}

/// Replace the first Python `pass` keyword with generated code, using word-boundary awareness
/// to avoid replacing substrings like `password`, `passenger`, `bypass`, etc.
fn replace_python_pass(prompt: &str, generated_code: &str) -> String {
    let mut replaced = false;
    let result = prompt
        .lines()
        .map(|line| {
            if !replaced {
                let trimmed = line.trim();
                if trimmed == "pass" {
                    replaced = true;
                    let indent = line.len() - trimmed.len();
                    let indent_str = " ".repeat(indent);
                    return generated_code
                        .lines()
                        .map(|code_line| format!("{}{}", indent_str, code_line.trim_start()))
                        .collect::<Vec<_>>()
                        .join("\n");
                }
            }
            line.to_string()
        })
        .collect::<Vec<_>>()
        .join("\n");
    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn extract_code_from_markdown_fence() {
        let response = "Here's the code:\n\n```python\ndef foo():\n    return 42\n```";
        let code = extract_code(response);
        assert!(code.contains("def foo()"));
        assert!(code.contains("return 42"));
        assert!(!code.contains("```"));
    }

    #[test]
    fn extract_code_without_fence() {
        let response = "def foo():\n    return 42";
        let code = extract_code(response);
        assert_eq!(code, "def foo():\n    return 42");
    }

    #[test]
    fn extract_code_empty_response() {
        let code = extract_code("");
        assert_eq!(code, "");
    }

    #[test]
    fn extract_code_whitespace_only() {
        let code = extract_code("   \n  ");
        assert_eq!(code, "");
    }

    #[test]
    fn replace_python_pass_preserves_password() {
        let prompt = "def login(password, bypass=True):\n    pass\n";
        let code = "    return True";
        let result = replace_python_pass(prompt, code);
        assert!(result.contains("password"));
        assert!(result.contains("bypass"));
        assert!(result.contains("return True"));
        assert!(!result.contains("    pass"));
    }

    #[test]
    fn replace_python_pass_simple() {
        let prompt = "def foo():\n    pass\n";
        let code = "    return 42";
        let result = replace_python_pass(prompt, code);
        assert_eq!(result, "def foo():\n    return 42");
    }

    #[test]
    fn replace_python_pass_replaces_only_first() {
        let prompt = "class Foo:\n    pass\n\nclass Bar:\n    pass\n";
        let code = "    return 42";
        let result = replace_python_pass(prompt, code);
        assert!(
            result.contains("return 42"),
            "First pass should be replaced"
        );
        // Second `pass` should remain
        let lines: Vec<&str> = result.lines().collect();
        assert!(
            lines.iter().any(|l| l.trim() == "pass"),
            "Second pass should remain"
        );
    }

    #[test]
    fn language_codes_contain_cs_and_rs() {
        let cs = LANGUAGE_CODES.iter().any(|(code, _)| *code == "cs");
        let rs = LANGUAGE_CODES.iter().any(|(code, _)| *code == "rs");
        assert!(cs, "C# should be in language codes");
        assert!(rs, "Rust should be in language codes");
    }

    #[test]
    fn default_config_uses_all_languages() {
        let config = yaml_serde::Value::Null;
        let parsed = parse_config(&config).unwrap();
        let all_codes: Vec<_> = LANGUAGE_CODES
            .iter()
            .map(|(c, _)| (*c).to_string())
            .collect();
        for code in &all_codes {
            assert!(
                parsed.languages.contains(code),
                "default should include '{}'",
                code
            );
        }
    }

    #[test]
    fn timeout_secs_capped_at_600() {
        let config: yaml_serde::Value = yaml_serde::from_str(
            r#"
            languages: [python]
            timeout_secs: 999999
            "#,
        )
        .unwrap();
        let parsed = parse_config(&config).unwrap();
        assert!(
            parsed.timeout_secs <= 600,
            "timeout_secs should be capped at 600"
        );
    }

    #[test]
    fn malformed_extract_code() {
        assert_eq!(extract_code(""), "");
        assert_eq!(extract_code("   "), "");
        assert_eq!(extract_code("just text"), "just text");
    }

    #[test]
    fn config_language_filter() {
        let config: yaml_serde::Value = yaml_serde::from_str(
            r#"
            languages: [go, java]
            num_samples: 10
            timeout_secs: 60
            "#,
        )
        .unwrap();
        let parsed = parse_config(&config).unwrap();
        assert_eq!(parsed.languages, vec!["go".to_string(), "java".to_string()]);
        assert_eq!(parsed.num_samples, Some(10));
        assert_eq!(parsed.timeout_secs, 60);
    }

    #[test]
    fn extract_code_nested_fences() {
        // A code block containing a fence line inside should stop at the first closing fence
        let response = "```python\ndef foo():\n    print('```')\n    return 42\n```";
        let code = extract_code(response);
        // Should extract up to the first closing fence
        assert!(code.contains("def foo()"));
    }

    #[test]
    fn extract_code_unclosed_fence() {
        // When there's an opening fence but no closing fence,
        // should strip the opening fence and return the code content
        let response = "```python\ndef foo():\n    return 42";
        let code = extract_code(response);
        // Opening fence and language identifier should be stripped
        assert!(!code.contains("```"));
        assert!(code.contains("def foo()"));
        assert!(code.contains("return 42"));
    }

    #[test]
    fn num_samples_bounds_check() {
        // Negative values should be filtered out
        let config: yaml_serde::Value = yaml_serde::from_str(
            r#"
            num_samples: -5
            "#,
        )
        .unwrap();
        let parsed = parse_config(&config).unwrap();
        assert_eq!(parsed.num_samples, None);

        // Valid values should pass through
        let config: yaml_serde::Value = yaml_serde::from_str(
            r#"
            num_samples: 10
            "#,
        )
        .unwrap();
        let parsed = parse_config(&config).unwrap();
        assert_eq!(parsed.num_samples, Some(10));

        // Values above 10K should be rejected
        let config: yaml_serde::Value = yaml_serde::from_str(
            r#"
            num_samples: 10001
            "#,
        )
        .unwrap();
        let parsed = parse_config(&config).unwrap();
        assert_eq!(parsed.num_samples, None, "10001 should exceed cap");
    }
}
