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

pub struct NQOpenBenchmark {
    state: Mutex<NQOpenState>,
}

struct NQOpenState {
    items: Vec<NQOpenItem>,
    current_idx: usize,
}

impl Default for NQOpenBenchmark {
    fn default() -> Self {
        Self {
            state: Mutex::new(NQOpenState {
                items: Vec::new(),
                current_idx: 0,
            }),
        }
    }
}

#[derive(Debug, Clone, Deserialize)]
struct NQOpenItem {
    question: String,
    answers: Vec<String>,
}

fn load_nq_open() -> Vec<NQOpenItem> {
    let cache_dir = dirs::cache_dir()
        .unwrap_or_default()
        .join("llm-benchmark-runner")
        .join("nq_open");
    let path = cache_dir.join("NQ-Open.json");

    if path.exists() {
        let content = fs::read_to_string(&path).expect("Failed to read cached NQ-Open");
        return serde_json::from_str(&content).expect("Failed to parse NQ-Open");
    }

    fs::create_dir_all(&cache_dir).expect("Failed to create cache dir");
    println!("  Downloading NQ-Open dataset...");
    let url = "https://huggingface.co/datasets/nq_open/resolve/main/data/test.csv";
    let bytes = download_with_retry_bytes(url, 3, 60, "llm-benchmark-runner")
        .expect("Failed to download NQ-Open");

    let content = String::from_utf8(Vec::from(bytes.as_ref())).expect("Failed to decode UTF-8");
    let mut reader = csv::ReaderBuilder::new()
        .delimiter(b',')
        .has_headers(true)
        .from_reader(content.as_bytes());
    let mut items = Vec::new();
    for record in reader.records().flatten() {
        if record.len() >= 2 {
            let question = record.get(0).unwrap_or("").to_string();
            let answers_str = record.get(1).unwrap_or("").to_string();
            let answers = answers_str
                .split("\\t")
                .map(|s| s.trim().to_string())
                .filter(|s| !s.is_empty())
                .collect();
            items.push(NQOpenItem { question, answers });
        }
    }

    fs::write(&path, &bytes).expect("Failed to save NQ-Open");
    items
}

impl Benchmark for NQOpenBenchmark {
    fn name(&self) -> &str {
        "nq_open"
    }

    fn display_name(&self) -> &'static str {
        "NQ Open (Natural Questions)"
    }

    fn category(&self) -> BenchmarkCategory {
        BenchmarkCategory::Knowledge
    }

    fn pre_execute(&self, _config: &yaml_serde::Value) -> Result<()> {
        let items = load_nq_open();
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
            "You are a knowledgeable assistant. Answer the question with a single entity (person, place, organization, etc.) or a short factual phrase.";

        let user_prompt = r#"Question: Who is the CEO of Tesla?
Answer: Elon Musk

Question: What is the capital of France?
Answer: Paris

Question: {question}
Answer:"#;

        let prompt = user_prompt.replace("{question}", &item.question);
        let response = tracker.chat_completion(&model.model_name, system_prompt, &prompt)?;

        let response = response.trim();
        let is_correct = item.answers.iter().any(|a| {
            let a = a.trim().to_lowercase();
            response.to_lowercase() == a
        });

        Ok(Some(
            TaskResult::new(
                format!("task-{}", idx),
                is_correct,
                if is_correct { 1.0 } else { 0.0 },
                vec![],
            )
            .with_metadata(Some(serde_json::json!({
                "question": item.question,
                "expected": item.answers,
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

        let em_score = if total > 0 {
            correct as f64 / total as f64
        } else {
            0.0
        };

        let mut scores = BTreeMap::new();
        scores.insert(
            "exact_match".to_string(),
            Score::float(em_score * 100.0, ScoreUnit::Percent)
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
                    "NQ Open: {} exact matches out of {} ({:.1}%)",
                    correct,
                    total,
                    em_score * 100.0
                ),
            }],
            raw: raw.clone(),
        })
    }
}
