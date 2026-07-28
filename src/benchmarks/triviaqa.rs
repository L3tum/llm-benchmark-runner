use crate::benchmarks::Benchmark;
use crate::config::Model;
use crate::download::download_with_retry_bytes;
use crate::reports::model::{BenchmarkCategory, BenchmarkResult, Score, ScoreUnit, TaskResult};
use crate::token_tracker::TokenTracker;
use anyhow::Result;
use serde::Deserialize;
use std::collections::BTreeMap;
use std::fs;
use std::sync::Mutex;

pub struct TriviaQABenchmark {
    state: Mutex<TriviaQAState>,
}

struct TriviaQAState {
    items: Vec<TriviaQARow>,
    current_idx: usize,
}

impl Default for TriviaQABenchmark {
    fn default() -> Self {
        Self {
            state: Mutex::new(TriviaQAState {
                items: Vec::new(),
                current_idx: 0,
            }),
        }
    }
}

#[derive(Debug, Clone, Deserialize)]
struct TriviaQARow {
    question: String,
    entity_pages: Option<Vec<TriviaQAEntity>>,
    question_source: Option<String>,
}

#[derive(Debug, Clone, Deserialize)]
struct TriviaQAEntity {
    title: String,
}

fn load_trivia_qa() -> Vec<TriviaQARow> {
    let cache_dir = dirs::cache_dir()
        .unwrap_or_default()
        .join("llm-benchmark-runner")
        .join("trivia_qa");
    let path = cache_dir.join("rc.nocontext.json");

    if path.exists() {
        let content = fs::read_to_string(&path).expect("Failed to read cached TriviaQA");
        return serde_json::from_str(&content).expect("Failed to parse TriviaQA");
    }

    fs::create_dir_all(&cache_dir).expect("Failed to create cache dir");
    println!("  Downloading TriviaQA (rc.nocontext) dataset...");
    let url =
        "https://huggingface.co/datasets/mandarjoshi/trivia_qa/resolve/main/rc/nocontext.json";
    let bytes = download_with_retry_bytes(url, 3, 60, "llm-benchmark-runner")
        .expect("Failed to download TriviaQA");

    let dataset: TriviaQADataset = serde_json::from_slice(&bytes).unwrap();
    let rows = dataset.data;
    fs::write(&path, &bytes).expect("Failed to save TriviaQA");
    rows
}

#[derive(Debug, Deserialize)]
struct TriviaQADataset {
    data: Vec<TriviaQARow>,
}

impl Benchmark for TriviaQABenchmark {
    fn name(&self) -> &str {
        "triviaqa"
    }

    fn display_name(&self) -> &'static str {
        "TriviaQA (Closed-Book)"
    }

    fn category(&self) -> BenchmarkCategory {
        BenchmarkCategory::Knowledge
    }

    fn pre_execute(&self, _config: &yaml_serde::Value) -> Result<()> {
        let items = load_trivia_qa();
        let mut state = self.state.lock().unwrap();
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
            let mut state = self.state.lock().unwrap();
            if state.current_idx >= state.items.len() {
                return Ok(None);
            }
            let idx = state.current_idx;
            let item = state.items[idx].clone();
            state.current_idx += 1;
            (item, idx)
        };

        let system_prompt =
            "You are a trivia expert. Answer with a single factual entity (person, place, thing).";

        let user_prompt = r#"Question: Who discovered penicillin?
Answer: Alexander Fleming

Question: What is the largest ocean on Earth?
Answer: Pacific Ocean

Question: {question}
Answer:"#;

        let prompt = user_prompt.replace("{question}", &item.question);
        let response = tracker.chat_completion(&model.model_name, system_prompt, &prompt)?;

        let response = response.trim().to_lowercase();
        let is_correct = item
            .entity_pages
            .as_ref()
            .map(|entities| {
                entities.iter().any(|e| {
                    let title = e.title.to_lowercase();
                    response.contains(&title) || title.contains(&response)
                })
            })
            .unwrap_or(false);

        let entity_titles: Vec<String> = item
            .entity_pages
            .as_ref()
            .map(|entities| entities.iter().map(|e| e.title.clone()).collect())
            .unwrap_or_default();

        Ok(Some(
            TaskResult::new(
                format!("task-{}", idx),
                is_correct,
                if is_correct { 1.0 } else { 0.0 },
                vec![],
            )
            .with_metadata(Some(serde_json::json!({
                "question": item.question,
                "entities": entity_titles,
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
                    "TriviaQA: {} correct out of {} ({:.1}%)",
                    correct,
                    total,
                    em_score * 100.0
                ),
            }],
            raw: raw.clone(),
        })
    }
}
