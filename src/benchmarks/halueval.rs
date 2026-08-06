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

pub struct HaluEvalBenchmark {
    state: Mutex<HaluEvalState>,
}

#[derive(Clone, serde::Serialize, serde::Deserialize)]
struct HaluEvalState {
    items: Vec<HaluEvalItem>,
    current_idx: usize,
}

impl Default for HaluEvalBenchmark {
    fn default() -> Self {
        Self {
            state: Mutex::new(HaluEvalState {
                items: Vec::new(),
                current_idx: 0,
            }),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct HaluEvalItem {
    question: String,
    context: String,
    answer: String,
    label: String, // "hallucinated" or "not_hallucinated" or similar
}

fn load_halueval_dataset() -> Vec<HaluEvalItem> {
    let cache_dir = dirs::cache_dir()
        .unwrap_or_default()
        .join("llm-benchmark-runner")
        .join("halueval");
    let path = cache_dir.join("qa.json");
    let url =
        "https://huggingface.co/datasets/jzjiao/halueval-sft/resolve/main/data/test-00000-of-00001-af0f10a1c83a1f93.parquet";

    if path.exists() {
        let content = fs::read_to_string(&path).expect("Failed to read cached HaluEval");
        return serde_json::from_str(&content).expect("Failed to parse HaluEval");
    }

    fs::create_dir_all(&cache_dir).expect("Failed to create cache dir");
    println!("  Downloading HaluEval (SFT test split) dataset...");
    let rows = download_parquet_records(url, 3, 60, "llm-benchmark-runner")
        .expect("Failed to download HaluEval");
    let mut items = Vec::new();
    for r in rows {
        let input = r
            .get("input")
            .and_then(|v| v.as_str())
            .unwrap_or("")
            .to_string();
        let label_raw = r
            .get("ground_truth_output")
            .and_then(|v| v.as_str())
            .unwrap_or("")
            .to_lowercase();
        if input.trim().is_empty() {
            continue;
        }
        let label = if label_raw.contains("inconsistent") {
            "hallucinated"
        } else {
            "not_hallucinated"
        };
        items.push(HaluEvalItem {
            question: String::new(),
            context: input,
            answer: String::new(),
            label: label.to_string(),
        });
    }
    fs::write(
        &path,
        serde_json::to_vec(&items).expect("Failed to save HaluEval"),
    )
    .expect("Failed to save HaluEval");
    items
}

impl Benchmark for HaluEvalBenchmark {
    fn name(&self) -> &str {
        "halueval"
    }

    fn display_name(&self) -> &'static str {
        "HaluEval (QA)"
    }

    fn category(&self) -> BenchmarkCategory {
        BenchmarkCategory::Hallucination
    }

    fn pre_execute(&self, _config: &yaml_serde::Value) -> Result<()> {
        let items = load_halueval_dataset();
        let mut state = self.state.lock().unwrap_or_else(|p| p.into_inner());
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
            let mut state = self.state.lock().unwrap_or_else(|p| p.into_inner());
            if state.current_idx >= state.items.len() {
                return Ok(None);
            }
            let idx = state.current_idx;
            let item = state.items[idx].clone();
            state.current_idx += 1;
            (item, idx)
        };

        let system_prompt =
            "You are a factuality evaluator. Given a question, context, and answer, determine if the answer contains hallucinated information (facts not supported by the context). Respond with 'hallucinated' or 'not_hallucinated'.";

        let user_prompt = r#"Question: What did the cat do?
Context: The cat was sleeping on the sofa.
Answer: The cat was sleeping on the sofa.
Verdict: not_hallucinated

Question: What did the cat do?
Context: The cat was sleeping on the sofa.
Answer: The cat was playing with a ball.
Verdict: hallucinated

Question: {question}
Context: {context}
Answer: {answer}
Verdict:"#;

        let prompt = user_prompt
            .replace("{question}", &item.question)
            .replace("{context}", &item.context)
            .replace("{answer}", &item.answer);

        let response = tracker.chat_completion(&model.model_name, system_prompt, &prompt)?;

        let response = response.trim().to_lowercase();
        let is_correct = match item.label.as_str() {
            "hallucinated" => response.contains("hallucinated"),
            "not_hallucinated" => response.contains("not_hallucinated"),
            _ => response.contains("not_hallucinated"),
        };

        Ok(Some(
            TaskResult::new(
                format!("task-{}", idx),
                is_correct,
                if is_correct { 1.0 } else { 0.0 },
                vec![],
            )
            .with_metadata(Some(serde_json::json!({
                "question": item.question,
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
                    "HaluEval QA: {}/{} correct ({:.1}%)",
                    correct,
                    total,
                    accuracy * 100.0
                ),
            }],
            raw: raw.clone(),
        })
    }
}
