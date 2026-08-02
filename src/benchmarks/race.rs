use crate::benchmarks::Benchmark;
use crate::config::Model;
use crate::download::download_with_retry_bytes;
use crate::shared::{BenchmarkCategory, BenchmarkResult, Score, ScoreUnit, TaskResult};
use crate::token_tracker::TokenTracker;
use anyhow::Result;
use serde::Deserialize;
use std::collections::BTreeMap;
use std::fs;
use std::sync::Mutex;

pub struct RaceBenchmark {
    state: Mutex<RaceState>,
}

struct RaceState {
    items: Vec<RaceItem>,
    current_idx: usize,
}

impl Default for RaceBenchmark {
    fn default() -> Self {
        Self {
            state: Mutex::new(RaceState {
                items: Vec::new(),
                current_idx: 0,
            }),
        }
    }
}

#[derive(Debug, Clone, Deserialize)]
struct RaceItem {
    passage: String,
    question: String,
    option_a: String,
    option_b: String,
    option_c: String,
    option_d: String,
    answer: String,
}

fn load_race_dataset() -> Vec<RaceItem> {
    let cache_dir = dirs::cache_dir()
        .unwrap_or_default()
        .join("llm-benchmark-runner")
        .join("race");
    let path = cache_dir.join("test.csv");

    if path.exists() {
        let content = fs::read_to_string(&path).expect("Failed to read cached RACE");
        let mut reader = csv::ReaderBuilder::new()
            .delimiter(b',')
            .has_headers(true)
            .from_reader(content.as_bytes());
        let mut items = Vec::new();
        for record in reader.records().flatten() {
            if record.len() >= 7 {
                items.push(RaceItem {
                    passage: record.get(0).unwrap_or("").to_string(),
                    question: record.get(1).unwrap_or("").to_string(),
                    option_a: record.get(2).unwrap_or("").to_string(),
                    option_b: record.get(3).unwrap_or("").to_string(),
                    option_c: record.get(4).unwrap_or("").to_string(),
                    option_d: record.get(5).unwrap_or("").to_string(),
                    answer: record.get(6).unwrap_or("").to_string(),
                });
            }
        }
        return items;
    }

    fs::create_dir_all(&cache_dir).expect("Failed to create cache dir");
    println!("  Downloading RACE dataset...");
    let url = "https://huggingface.co/datasets/ehovy/race/resolve/main/test.csv";
    let bytes = download_with_retry_bytes(url, 3, 60, "llm-benchmark-runner")
        .expect("Failed to download RACE");

    let content = String::from_utf8(Vec::from(bytes.as_ref())).expect("Failed to decode UTF-8");
    let mut reader = csv::ReaderBuilder::new()
        .delimiter(b',')
        .has_headers(true)
        .from_reader(content.as_bytes());
    let mut items = Vec::new();
    for record in reader.records().flatten() {
        if record.len() >= 7 {
            items.push(RaceItem {
                passage: record.get(0).unwrap_or("").to_string(),
                question: record.get(1).unwrap_or("").to_string(),
                option_a: record.get(2).unwrap_or("").to_string(),
                option_b: record.get(3).unwrap_or("").to_string(),
                option_c: record.get(4).unwrap_or("").to_string(),
                option_d: record.get(5).unwrap_or("").to_string(),
                answer: record.get(6).unwrap_or("").to_string(),
            });
        }
    }

    fs::write(&path, &bytes).expect("Failed to save RACE");
    items
}

impl Benchmark for RaceBenchmark {
    fn name(&self) -> &str {
        "race"
    }

    fn display_name(&self) -> &'static str {
        "RACE (Reading Comprehension)"
    }

    fn category(&self) -> BenchmarkCategory {
        BenchmarkCategory::Hallucination
    }

    fn pre_execute(&self, _config: &yaml_serde::Value) -> Result<()> {
        let items = load_race_dataset();
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

        let system_prompt = "You are a reading comprehension expert. Read the passage carefully and answer the multiple-choice question based ONLY on the passage. Do not use external knowledge.";

        let user_prompt = r#"Passage: The cat sat on the mat.
Question: What did the cat sit on?
A. The bed
B. The mat
C. The table
D. The chair
Answer: B

Passage: {passage}
Question: {question}
A. {option_a}
B. {option_b}
C. {option_c}
D. {option_d}
Answer:"#;

        let prompt = user_prompt
            .replace("{passage}", &item.passage)
            .replace("{question}", &item.question)
            .replace("{option_a}", &item.option_a)
            .replace("{option_b}", &item.option_b)
            .replace("{option_c}", &item.option_c)
            .replace("{option_d}", &item.option_d);

        let response = tracker.chat_completion(&model.model_name, system_prompt, &prompt)?;

        let response = response.trim().to_ascii_uppercase();
        let is_correct = response.starts_with(&item.answer.to_ascii_uppercase());

        Ok(Some(
            TaskResult::new(
                format!("task-{}", idx),
                is_correct,
                if is_correct { 1.0 } else { 0.0 },
                vec![],
            )
            .with_metadata(Some(serde_json::json!({
                "expected": item.answer,
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
                    "RACE (Reading Comprehension — Hallucination category): {}/{} correct ({:.1}%)",
                    correct,
                    total,
                    accuracy * 100.0
                ),
            }],
            raw: raw.clone(),
        })
    }
}
