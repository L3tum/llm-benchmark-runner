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

pub struct FaithDialBenchmark {
    state: Mutex<FaithDialState>,
}

struct FaithDialState {
    items: Vec<FaithDialItem>,
    current_idx: usize,
}

impl Default for FaithDialBenchmark {
    fn default() -> Self {
        Self {
            state: Mutex::new(FaithDialState {
                items: Vec::new(),
                current_idx: 0,
            }),
        }
    }
}

#[derive(Debug, Clone, Deserialize)]
struct FaithDialItem {
    context: String,  // dialogue with background knowledge
    response: String, // assistant response
    label: String,    // "1" (hallucinated) or "0" (faithful)
}

fn load_faithdial_dataset() -> Vec<FaithDialItem> {
    let cache_dir = dirs::cache_dir()
        .unwrap_or_default()
        .join("llm-benchmark-runner")
        .join("faithdial");
    let path = cache_dir.join("faithdial.json");

    if path.exists() {
        let content = fs::read_to_string(&path).expect("Failed to read cached FaithDial");
        return serde_json::from_str(&content).expect("Failed to parse FaithDial");
    }

    fs::create_dir_all(&cache_dir).expect("Failed to create cache dir");
    println!("  Downloading FaithDial dataset...");
    let url = "https://huggingface.co/datasets/dataframer/faithdial/resolve/main/faithdial.json";
    let bytes = download_with_retry_bytes(url, 3, 60, "llm-benchmark-runner")
        .expect("Failed to download FaithDial");

    let items: Vec<FaithDialItem> =
        serde_json::from_slice(&bytes).expect("Failed to parse FaithDial");
    fs::write(&path, bytes).expect("Failed to save FaithDial");
    items
}

impl Benchmark for FaithDialBenchmark {
    fn name(&self) -> &str {
        "faithdial"
    }

    fn display_name(&self) -> &'static str {
        "FaithDial"
    }

    fn category(&self) -> BenchmarkCategory {
        BenchmarkCategory::Hallucination
    }

    fn pre_execute(&self, _config: &yaml_serde::Value) -> Result<()> {
        let items = load_faithdial_dataset();
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
            "You are a factuality evaluator for dialogues. Given a dialogue with background knowledge and an assistant response, determine if the response hallucinates information not supported by the dialogue. Respond with 'hallucinated' or 'not_hallucinated'.";

        let user_prompt = r#"Dialogue: What's the weather like today?
Response: It's sunny.
Label: not_hallucinated

Dialogue: I booked a flight to Paris.
Response: Your flight is to London.
Label: hallucinated

Dialogue: I'm going to the gym at 5pm.
Response: You're going to the gym.
Label: not_hallucinated

Dialogue: My mother's name is Jane.
Response: Your mother's name is Mary.
Label: hallucinated

Dialogue: I like chocolate.
Response: You like vanilla.
Label: hallucinated

Dialogue: The meeting is at 3pm.
Response: The meeting is at 3pm.
Label: not_hallucinated

Dialogue: I work as a teacher.
Response: You work as an engineer.
Label: hallucinated

Dialogue: The capital of France is Paris.
Response: The capital of France is Paris.
Label: not_hallucinated

Now, evaluate the following:

Dialogue: {context}
Response: {response}
Label:"#;

        let prompt = user_prompt
            .replace("{context}", &item.context)
            .replace("{response}", &item.response);

        let response = tracker.chat_completion(&model.model_name, system_prompt, &prompt)?;

        let response = response.trim().to_lowercase();
        let is_correct = match item.label.as_str() {
            "1" => response.contains("hallucinated"),
            "0" => response.contains("not_hallucinated"),
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
                    "FaithDial: {}/{} correct ({:.1}%)",
                    correct,
                    total,
                    accuracy * 100.0
                ),
            }],
            raw: raw.clone(),
        })
    }
}
