use crate::benchmarks::Benchmark;
use crate::config::Model;
use crate::download::download_parquet_records;
use crate::shared::{BenchmarkCategory, BenchmarkResult, Score, ScoreUnit, TaskResult};
use crate::token_tracker::TokenTracker;
use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::fs;
use std::sync::Mutex;

pub struct TrueFalseBenchmark {
    state: Mutex<TrueFalseState>,
}

struct TrueFalseState {
    items: Vec<TrueFalseItem>,
    current_idx: usize,
}

impl Default for TrueFalseBenchmark {
    fn default() -> Self {
        Self {
            state: Mutex::new(TrueFalseState {
                items: Vec::new(),
                current_idx: 0,
            }),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct TrueFalseItem {
    statement: String,
    label: String, // "True", "False"
}

fn load_true_false_dataset() -> Vec<TrueFalseItem> {
    let cache_dir = dirs::cache_dir()
        .unwrap_or_default()
        .join("llm-benchmark-runner")
        .join("true_false");
    let path = cache_dir.join("statements.json");
    let url =
        "https://huggingface.co/datasets/truthfulqa/truthful_qa/resolve/main/generation/validation-00000-of-00001.parquet";

    if path.exists() {
        let content = fs::read_to_string(&path).expect("Failed to read cached True-False dataset");
        return serde_json::from_str(&content).expect("Failed to parse True-False dataset");
    }

    fs::create_dir_all(&cache_dir).expect("Failed to create cache dir");
    println!("  Downloading True-False dataset from TruthfulQA...");

    let rows = download_parquet_records(url, 3, 60, "llm-benchmark-runner")
        .expect("Failed to download TruthfulQA generation dataset");
    let mut items = Vec::new();
    for r in rows {
        // TruthfulQA `best_answer` is the verified-true answer; `incorrect_answers`
        // are the false misconceptions. Use those as the True/False statements.
        if let Some(ba) = r.get("best_answer").and_then(|v| v.as_str()) {
            if !ba.trim().is_empty() {
                items.push(TrueFalseItem {
                    statement: ba.to_string(),
                    label: "True".to_string(),
                });
            }
        }
        if let Some(fa) = r.get("incorrect_answers").and_then(|v| v.as_array()) {
            for a in fa.iter().filter_map(|x| x.as_str()) {
                if !a.trim().is_empty() {
                    items.push(TrueFalseItem {
                        statement: a.to_string(),
                        label: "False".to_string(),
                    });
                }
            }
        }
    }
    fs::write(
        &path,
        serde_json::to_vec(&items).expect("Failed to save True-False dataset"),
    )
    .expect("Failed to save True-False dataset");
    items
}

impl Benchmark for TrueFalseBenchmark {
    fn name(&self) -> &str {
        "true_false"
    }

    fn display_name(&self) -> &'static str {
        "True-False"
    }

    fn category(&self) -> BenchmarkCategory {
        BenchmarkCategory::Hallucination
    }

    fn pre_execute(&self, _config: &yaml_serde::Value) -> Result<()> {
        let items = load_true_false_dataset();
        let mut state = self.state.lock().expect(crate::shared::MUTEX_PANIC_MSG);
        state.items = items;
        state.current_idx = 0;
        Ok(())
    }

    fn execute_one(
        &self,
        model: &Model,
        _config: &yaml_serde::Value,
        tracker: &mut TokenTracker,
    ) -> Result<Option<TaskResult>> {
        let (item, idx) = {
            let mut state = self.state.lock().expect(crate::shared::MUTEX_PANIC_MSG);
            if state.current_idx >= state.items.len() {
                return Ok(None);
            }
            let idx = state.current_idx;
            let item = state.items[idx].clone();
            state.current_idx += 1;
            (item, idx)
        };

        let system_prompt =
            "You are a factuality checker. Given a statement, determine if it is True or False based on your knowledge. Respond with only 'True' or 'False'.";

        let user_prompt = r#"Statement: The capital of Australia is Sydney.
Answer: False

Statement: The Earth revolves around the Sun.
Answer: True

Statement: Humans have 46 chromosomes.
Answer: False

Statement: Albert Einstein was born in 1879.
Answer: True

Statement: The moon is made of cheese.
Answer: False

Statement: Water boils at 100 degrees Celsius at sea level.
Answer: True

Statement: The Eiffel Tower is in Paris.
Answer: True

Statement: The sun rises in the west.
Answer: False

Statement: {statement}
Answer:"#;

        let prompt = user_prompt.replace("{statement}", &item.statement);
        let response = tracker.chat_completion(&model.model_name, system_prompt, &prompt)?;

        let response = response.trim().to_lowercase();
        let is_correct = match item.label.to_lowercase().as_str() {
            "true" => response.contains("true"),
            "false" => response.contains("false"),
            _ => response.contains("false"),
        };

        Ok(Some(
            TaskResult::new(
                format!("task-{}", idx),
                is_correct,
                if is_correct { 1.0 } else { 0.0 },
                vec![],
            )
            .with_metadata(Some(serde_json::json!({
                "statement": item.statement,
                "expected": item.label,
                "response": response,
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
                    "True-False: {}/{} correct ({:.1}%)",
                    correct,
                    total,
                    accuracy * 100.0
                ),
            }],
            raw: raw.clone(),
        })
    }
}
