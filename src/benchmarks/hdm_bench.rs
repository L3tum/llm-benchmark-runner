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

pub struct HdmBenchBenchmark {
    state: Mutex<HdmBenchState>,
}

struct HdmBenchState {
    items: Vec<HDMItem>,
    current_idx: usize,
}

impl Default for HdmBenchBenchmark {
    fn default() -> Self {
        Self {
            state: Mutex::new(HdmBenchState {
                items: Vec::new(),
                current_idx: 0,
            }),
        }
    }
}

#[derive(Debug, Clone, Deserialize)]
struct HDMItem {
    prompt: String,
    context: String,
    response: String,
    label: String, // "hallucinated", "grounded", or "common_knowledge"
}

fn load_hdm_dataset() -> Vec<HDMItem> {
    let cache_dir = dirs::cache_dir()
        .unwrap_or_default()
        .join("llm-benchmark-runner")
        .join("hdm_bench");
    let path = cache_dir.join("HDM-Bench.json");

    if path.exists() {
        let content = fs::read_to_string(&path).expect("Failed to read cached HDM-Bench");
        return serde_json::from_str(&content).expect("Failed to parse HDM-Bench");
    }

    fs::create_dir_all(&cache_dir).expect("Failed to create cache dir");
    println!("  Downloading HDM-Bench synthetic dataset...");
    let url = "https://huggingface.co/datasets/dataframer/HDM-Bench/resolve/main/HDM-Bench.json";
    let bytes = download_with_retry_bytes(url, 3, 60, "llm-benchmark-runner")
        .expect("Failed to download HDM-Bench");

    let items: Vec<HDMItem> = serde_json::from_slice(&bytes).expect("Failed to parse HDM-Bench");
    fs::write(&path, &bytes).expect("Failed to save HDM-Bench");
    items
}

impl Benchmark for HdmBenchBenchmark {
    fn name(&self) -> &str {
        "hdm_bench"
    }

    fn display_name(&self) -> &'static str {
        "HDM-Bench (Context Hallucination)"
    }

    fn category(&self) -> BenchmarkCategory {
        BenchmarkCategory::Hallucination
    }

    fn pre_execute(&self, _config: &yaml_serde::Value) -> Result<()> {
        let items = load_hdm_dataset();
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
            "You are a factuality evaluator. Given a prompt, context, and response, determine if the response is hallucinated (contains information not grounded in the context), grounded (fully supported by the context), or common knowledge (factual information not in the context but universally known). Respond with 'hallucinated', 'grounded', or 'common_knowledge'.";

        let user_prompt = r#"Prompt: What is the capital of France?
Context: [empty]
Response: Paris.
Label: grounded

Prompt: Who wrote this paper?
Context: This paper was written by Alice and Bob.
Response: Charlie wrote the paper.
Label: hallucinated

Prompt: What is the speed of light?
Context: The study discusses quantum entanglement.
Response: The speed of light is 299,792,458 m/s.
Label: common_knowledge

Prompt: What did Alice say?
Context: Alice said the meeting is at 3pm.
Response: Alice said the meeting is at 5pm.
Label: hallucinated

Prompt: {prompt}
Context: {context}
Response: {response}
Label:"#;

        let prompt = user_prompt
            .replace("{prompt}", &item.prompt)
            .replace("{context}", &item.context)
            .replace("{response}", &item.response);

        let response = tracker.chat_completion(&model.model_name, system_prompt, &prompt)?;

        let response = response.trim().to_lowercase();
        let expected = item.label.to_lowercase();
        let is_correct = if expected == "hallucinated" {
            response.contains("hallucinat")
        } else if expected == "grounded" {
            response.contains("grounded")
        } else if expected == "common_knowledge" {
            response.contains("common.knowledge")
                || response.contains("common-knowledge")
                || response.contains("common knowledge")
        } else {
            false
        };

        let categories = vec![item.label.clone()];

        Ok(Some(
            TaskResult::new(
                format!("task-{}", idx),
                is_correct,
                if is_correct { 1.0 } else { 0.0 },
                categories,
            )
            .with_metadata(Some(serde_json::json!({
                "prompt": item.prompt,
                "expected": item.label,
                "response": response,
            }))),
        ))
    }

    fn to_report_result(&self, b: &BenchmarkResult) -> Result<BenchmarkResult> {
        let raw = &b.raw;

        let (total, correct, output_tokens, thinking_tokens, label_stats) = {
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

                let mut label_stats: BTreeMap<String, (i64, i64)> = BTreeMap::new();
                for task in per_task {
                    if let Some(label) = task
                        .get("categories")
                        .and_then(|v| v.as_array())
                        .and_then(|arr| arr.first())
                        .and_then(|v| v.as_str())
                    {
                        let (p, t) = label_stats.entry(label.to_string()).or_insert((0, 0));
                        *t += 1;
                        if task
                            .get("passed")
                            .and_then(|v| v.as_bool())
                            .unwrap_or(false)
                        {
                            *p += 1;
                        }
                    }
                }
                (total, correct, out, think, label_stats)
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
                    BTreeMap::new(),
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

        // Label breakdown
        let mut breakdowns = BTreeMap::new();
        if !label_stats.is_empty() {
            let mut rows = BTreeMap::new();
            for (label, (label_correct, label_total)) in &label_stats {
                let rate = if *label_total > 0 {
                    *label_correct as f64 / *label_total as f64
                } else {
                    0.0
                };
                rows.insert(
                    label.clone(),
                    BTreeMap::from([
                        (
                            "accuracy".to_string(),
                            Score::float(rate, ScoreUnit::Percent)
                                .display(format!("{:.1}%", rate * 100.0)),
                        ),
                        (
                            "instances".to_string(),
                            Score::integer(*label_total, ScoreUnit::Count)
                                .display(format!("{}/{}", label_correct, label_total)),
                        ),
                    ]),
                );
            }
            breakdowns.insert(
                "By Label".to_string(),
                crate::reports::model::BreakdownTable {
                    title: "Accuracy by Label".to_string(),
                    rows,
                },
            );
        }

        Ok(BenchmarkResult {
            scores,
            breakdowns,
            error_classification: BTreeMap::new(),
            artifacts: vec![],
            diagnostics: vec![crate::reports::model::Diagnostic {
                level: "info".to_string(),
                message: format!(
                    "HDM-Bench: {}/{} correct ({:.1}%)",
                    correct,
                    total,
                    accuracy * 100.0
                ),
            }],
            raw: raw.clone(),
        })
    }
}
