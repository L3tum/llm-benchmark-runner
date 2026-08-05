use crate::benchmarks::Benchmark;
use crate::config::Model;
use crate::docker_runner::{DockerMount, DockerRunConfig, DockerRunner};
use crate::download::download_with_retry_bytes;
use crate::shared::{
    fence_prompt_value, BenchmarkCategory, BenchmarkResult, Score, ScoreUnit, TaskResult,
};
use crate::token_tracker::TokenTracker;
use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::fs;
use std::sync::Mutex;
use tempfile::NamedTempFile;

pub struct CruxEvalBenchmark {
    state: Mutex<CruxEvalState>,
}

struct CruxEvalState {
    items: Vec<CruxEvalItem>,
    current_idx: usize,
    subset: CruxEvalSubset,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum CruxEvalSubset {
    Input,  // Predict input from code + output
    Output, // Predict output from code + input
    Repair, // Find/repair bugs in code
}

impl Default for CruxEvalBenchmark {
    fn default() -> Self {
        Self {
            state: Mutex::new(CruxEvalState {
                items: Vec::new(),
                current_idx: 0,
                subset: CruxEvalSubset::Output,
            }),
        }
    }
}

#[derive(Debug, Clone, Deserialize, Serialize)]
struct CruxEvalItem {
    task_id: String,
    code: String,
    input: String,
    output: String,
    #[serde(default)]
    buggy_code: Option<String>,
    #[serde(default)]
    task_type: Option<String>,
}

fn load_cruxeval_dataset(_subset: CruxEvalSubset, max_items: usize) -> Result<Vec<CruxEvalItem>> {
    let cache_dir = dirs::cache_dir()
        .unwrap_or_default()
        .join("llm-benchmark-runner")
        .join("cruxeval");
    let path = cache_dir.join("cruxeval.jsonl");
    let url = "https://huggingface.co/datasets/cruxeval-org/cruxeval/resolve/main/test.jsonl";

    if path.exists() {
        let content =
            fs::read_to_string(&path).with_context(|| "Failed to read cached CRUXEval")?;
        let items: Vec<CruxEvalItem> = content
            .lines()
            .filter_map(|line| serde_json::from_str(line).ok())
            .collect();
        return Ok(items.into_iter().take(max_items).collect());
    }

    fs::create_dir_all(&cache_dir).expect("Failed to create cache dir");
    println!(
        "  Downloading CRUXEval test set (up to {} instances)...",
        max_items
    );

    // cruxeval-org/cruxeval ships a single test.jsonl with code/input/output/id.
    let bytes = download_with_retry_bytes(url, 3, 120, "llm-benchmark-runner")?;
    let content = String::from_utf8(bytes.to_vec()).expect("Failed to decode UTF-8");
    let items: Vec<CruxEvalItem> = content
        .lines()
        .filter_map(|line| {
            serde_json::from_str::<serde_json::Value>(line)
                .ok()
                .map(|v| CruxEvalItem {
                    task_id: v
                        .get("id")
                        .and_then(|x| x.as_str())
                        .unwrap_or("")
                        .to_string(),
                    code: v
                        .get("code")
                        .and_then(|x| x.as_str())
                        .unwrap_or("")
                        .to_string(),
                    input: v
                        .get("input")
                        .and_then(|x| x.as_str())
                        .unwrap_or("")
                        .to_string(),
                    output: v
                        .get("output")
                        .and_then(|x| x.as_str())
                        .unwrap_or("")
                        .to_string(),
                    buggy_code: None,
                    task_type: None,
                })
        })
        .take(max_items)
        .collect();
    fs::write(&path, &content).expect("Failed to save CRUXEval");
    Ok(items)
}

/// Python normalization script for CRUXEval evaluation (executed in Docker).
const CRUX_EVAL_SCRIPT: &str = r#"
import sys
import ast
import os

def normalize(s):
    s = s.strip()
    try:
        val = ast.literal_eval(s)
        if isinstance(val, (list, tuple)):
            return [normalize(str(x)) for x in val]
        return str(val).strip()
    except:
        return s.strip()

pred = normalize(os.environ["PREDICTED"])
gold = normalize(os.environ["GROUND_TRUTH"])

if pred == gold:
    sys.exit(0)
try:
    if abs(float(pred) - float(gold)) < 1e-6:
        sys.exit(0)
except:
    pass
try:
    pred_list = [float(x) for x in pred]
    gold_list = [float(x) for x in gold]
    if len(pred_list) == len(gold_list) and all(abs(a-b) < 1e-6 for a,b in zip(pred_list, gold_list)):
        sys.exit(0)
except:
    pass
sys.exit(1)
"#;

/// Python repair evaluation script (executed in Docker).
/// Executes the fixed code with test input and compares output against ground truth.
const CRUX_REPAIR_SCRIPT: &str = r#"
import os
import sys
import tempfile
import subprocess

code = os.environ["PREDICTED_CODE"]
test_input = os.environ["TEST_INPUT"]
ground_truth = os.environ["GROUND_TRUTH"].strip()

with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
    f.write(code)
    script_path = f.name

try:
    result = subprocess.run(
        [sys.executable, script_path],
        input=test_input,
        capture_output=True,
        text=True,
        timeout=10,
    )
    output = result.stdout.strip()
    if output == ground_truth:
        sys.exit(0)
    try:
        if abs(float(output) - float(ground_truth)) < 1e-6:
            sys.exit(0)
    except:
        pass
    sys.exit(1)
except Exception:
    sys.exit(1)
finally:
    os.unlink(script_path)
"#;

/// Evaluate predicted output against ground truth using Docker-isolated Python.
fn evaluate_prediction(predicted: &str, ground_truth: &str) -> bool {
    if predicted.trim() == ground_truth.trim() {
        return true;
    }
    match evaluate_prediction_docker(predicted, ground_truth) {
        Ok(result) => result,
        Err(e) => {
            eprintln!("  Docker eval unavailable ({e}), using exact match fallback");
            false
        }
    }
}

/// Evaluate prediction in a Docker container. Untrusted data passed via env vars only.
fn evaluate_prediction_docker(predicted: &str, ground_truth: &str) -> Result<bool> {
    let script_file = NamedTempFile::new().context("Failed to create temp script file")?;
    fs::write(script_file.path(), CRUX_EVAL_SCRIPT).context("Failed to write eval script")?;

    let mut config = DockerRunConfig::new(
        "python:3.11-slim",
        vec!["python3".to_string(), "/tmp/eval.py".to_string()],
        30,
    );
    config
        .mounts
        .push(DockerMount::readonly(script_file.path(), "/tmp/eval.py"));
    config
        .env
        .push(("PREDICTED".to_string(), predicted.to_string()));
    config
        .env
        .push(("GROUND_TRUTH".to_string(), ground_truth.to_string()));

    let result = DockerRunner::run(&config)?;
    Ok(result.success())
}

/// Evaluate repaired code by executing it in Docker with the test input.
fn evaluate_repair_docker(fixed_code: &str, test_input: &str, ground_truth: &str) -> Result<bool> {
    let script_file = NamedTempFile::new().context("Failed to create temp script file")?;
    fs::write(script_file.path(), CRUX_REPAIR_SCRIPT).context("Failed to write repair script")?;

    let mut config = DockerRunConfig::new(
        "python:3.11-slim",
        vec!["python3".to_string(), "/tmp/repair.py".to_string()],
        30,
    );
    config
        .mounts
        .push(DockerMount::readonly(script_file.path(), "/tmp/repair.py"));
    config
        .env
        .push(("PREDICTED_CODE".to_string(), fixed_code.to_string()));
    config
        .env
        .push(("TEST_INPUT".to_string(), test_input.to_string()));
    config
        .env
        .push(("GROUND_TRUTH".to_string(), ground_truth.to_string()));

    let result = DockerRunner::run(&config)?;
    Ok(result.success())
}

impl Benchmark for CruxEvalBenchmark {
    fn name(&self) -> &str {
        "cruxeval"
    }

    fn display_name(&self) -> &'static str {
        "CRUXEval (Code Understanding)"
    }

    fn category(&self) -> BenchmarkCategory {
        BenchmarkCategory::Reasoning
    }

    fn pre_execute(&self, config: &yaml_serde::Value) -> Result<()> {
        let subset = config
            .get("subset")
            .and_then(|v| v.as_str())
            .unwrap_or("output");
        let subset = match subset {
            "input" => CruxEvalSubset::Input,
            "repair" => CruxEvalSubset::Repair,
            _ => CruxEvalSubset::Output,
        };
        let max_items = config
            .get("max_items")
            .and_then(|v| v.as_u64())
            .unwrap_or(200) as usize;
        let items = load_cruxeval_dataset(subset, max_items)?;
        let subset_name = match subset {
            CruxEvalSubset::Input => "input",
            CruxEvalSubset::Output => "output",
            CruxEvalSubset::Repair => "repair",
        };
        println!(
            "  CRUXEval {}: {} instances loaded (max: {})",
            subset_name,
            items.len(),
            max_items
        );
        let mut state = self.state.lock().expect(crate::shared::MUTEX_PANIC_MSG);
        state.items = items;
        state.current_idx = 0;
        state.subset = subset;
        Ok(())
    }

    fn execute_one(
        &self,
        model: &Model,
        _config: &yaml_serde::Value,
        tracker: &mut TokenTracker,
    ) -> Result<Option<TaskResult>> {
        let (item, idx, subset) = {
            let mut state = self.state.lock().expect(crate::shared::MUTEX_PANIC_MSG);
            if state.current_idx >= state.items.len() {
                return Ok(None);
            }
            let idx = state.current_idx;
            let item = state.items[idx].clone();
            state.current_idx += 1;
            let subset = state.subset;
            (item, idx, subset)
        };

        let system_prompt = match subset {
            CruxEvalSubset::Output => {
                "You are a code understanding assistant. Given Python code and its input, predict the exact output when the code is executed. Output only the expected output, nothing else."
            }
            CruxEvalSubset::Input => {
                "You are a code understanding assistant. Given Python code and its output, predict what input was provided that produced this output. Output only the expected input, nothing else."
            }
            CruxEvalSubset::Repair => {
                "You are a code debugging assistant. Given buggy Python code, its input, and expected output, identify and fix the bug. Output only the corrected code."
            }
        };

        let user_prompt = match subset {
            CruxEvalSubset::Output => {
                r#"Predict the exact output of this Python code when executed.

Code:
```python
<code>{code}</code>
```

Input:
<input>{input}</input>

Output:
"#
            }
            CruxEvalSubset::Input => {
                r#"Predict the exact input that was provided to this Python code to produce the given output.

Code:
```python
<code>{code}</code>
```

Output:
<output>{output}</output>

Input:
"#
            }
            CruxEvalSubset::Repair => {
                r#"This Python code has a bug. Given the code, its input, and the expected output, fix the bug.

Buggy Code:
```python
<buggy_code>{buggy_code}</buggy_code>
```

Input:
<input>{input}</input>

Expected Output:
<output>{output}</output>

Fixed Code:
```python
"#
            }
        };

        let prompt = match subset {
            CruxEvalSubset::Output => user_prompt
                .replace("{code}", &fence_prompt_value(&item.code))
                .replace("{input}", &fence_prompt_value(&item.input)),
            CruxEvalSubset::Input => user_prompt
                .replace("{code}", &fence_prompt_value(&item.code))
                .replace("{output}", &fence_prompt_value(&item.output)),
            CruxEvalSubset::Repair => {
                let buggy = item.buggy_code.as_deref().unwrap_or(&item.code);
                user_prompt
                    .replace("{buggy_code}", &fence_prompt_value(buggy))
                    .replace("{input}", &fence_prompt_value(&item.input))
                    .replace("{output}", &fence_prompt_value(&item.output))
            }
        };

        let response = tracker.chat_completion(&model.model_name, system_prompt, &prompt)?;
        let response_trimmed = response.trim();

        // Evaluate correctness: Docker-isolated execution for all subsets
        let is_correct = match subset {
            CruxEvalSubset::Output | CruxEvalSubset::Input => {
                evaluate_prediction(response_trimmed, &item.output)
            }
            CruxEvalSubset::Repair => {
                evaluate_repair_docker(response_trimmed, &item.input, &item.output)
                    .unwrap_or_default()
            }
        };

        Ok(Some(
            TaskResult::new(
                format!("task-{}", idx),
                is_correct,
                if is_correct { 1.0 } else { 0.0 },
                vec![],
            )
            .with_metadata(Some(serde_json::json!({
                "task_id": item.task_id,
                "subset": match subset {
                    CruxEvalSubset::Input => "input",
                    CruxEvalSubset::Output => "output",
                    CruxEvalSubset::Repair => "repair",
                },
                "expected": if subset == CruxEvalSubset::Repair { item.input } else { item.output },
                "response": response_trimmed,
            }))),
        ))
    }

    fn to_report_result(&self, b: &BenchmarkResult) -> Result<BenchmarkResult> {
        let raw = &b.raw;

        let (total, correct, output_tokens, thinking_tokens) = {
            if let Some(per_task) = raw.get("per_task").and_then(|v| v.as_array()) {
                let total = per_task.len() as i64;
                let correct = per_task
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
                (total, correct, out, think)
            } else {
                (
                    raw.get("total").and_then(|v| v.as_i64()).unwrap_or(0),
                    raw.get("correct").and_then(|v| v.as_i64()).unwrap_or(0),
                    raw.get("output_tokens")
                        .and_then(|v| v.as_i64())
                        .unwrap_or(0),
                    raw.get("thinking_tokens")
                        .and_then(|v| v.as_i64())
                        .unwrap_or(0),
                )
            }
        };

        let accuracy = if total > 0 {
            correct as f64 / total as f64
        } else {
            0.0
        };

        let mut scores = BTreeMap::new();
        scores.insert(
            "accuracy".to_string(),
            Score::float(accuracy * 100.0, ScoreUnit::Percent)
                .primary(true)
                .higher_is_better(true),
        );
        scores.insert("total".to_string(), Score::integer(total, ScoreUnit::Count));
        scores.insert(
            "correct".to_string(),
            Score::integer(correct, ScoreUnit::Count),
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
            diagnostics: vec![crate::reports::model::Diagnostic {
                level: "info".to_string(),
                message: format!(
                    "CRUXEval: {}/{} correct ({:.1}%)",
                    correct,
                    total,
                    accuracy * 100.0
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
    fn test_evaluate_prediction_exact_match() {
        assert!(evaluate_prediction("hello", "hello"));
        assert!(evaluate_prediction("  hello  ", "hello"));
    }

    #[test]
    fn test_evaluate_prediction_no_match() {
        assert!(!evaluate_prediction("hello", "world"));
    }
}
