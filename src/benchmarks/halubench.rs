use crate::benchmarks::Benchmark;
use crate::config::Model;
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

pub struct HaluBenchBenchmark {
    state: Mutex<HalUBenchState>,
}

struct HalUBenchState {
    items: Vec<HalUBenchItem>,
    current_idx: usize,
}

impl Default for HaluBenchBenchmark {
    fn default() -> Self {
        Self {
            state: Mutex::new(HalUBenchState {
                items: Vec::new(),
                current_idx: 0,
            }),
        }
    }
}

#[derive(Debug, Clone, Deserialize, Serialize)]
struct HalUBenchItem {
    question: String,
    context: String,
    answer: String,
    #[serde(default)]
    label: String, // "true", "false", "unanswerable"
    #[serde(default)]
    dataset: String,
    #[serde(default)]
    task_id: String,
}

fn load_halubench_dataset(max_items: usize) -> Result<Vec<HalUBenchItem>> {
    let cache_dir = dirs::cache_dir()
        .unwrap_or_default()
        .join("llm-benchmark-runner")
        .join("halubench");
    let path = cache_dir.join("halubench.json");

    if path.exists() {
        let content = fs::read_to_string(&path).expect("Failed to read cached HaluBench");
        let items: Vec<HalUBenchItem> =
            serde_json::from_str(&content).context("Failed to parse HaluBench")?;
        return Ok(items.into_iter().take(max_items).collect());
    }

    fs::create_dir_all(&cache_dir).expect("Failed to create cache dir");
    println!(
        "  Downloading HaluBench dataset (up to {} instances)...",
        max_items
    );

    // HaluBench on HuggingFace
    let urls = [
        "https://huggingface.co/datasets/tianyi-lab/HaluBench/resolve/main/data.json",
        "https://huggingface.co/datasets/tianyi-lab/HaluBench/resolve/main/halubench.json",
        "https://huggingface.co/datasets/tianyi-lab/HaluBench/resolve/main/test.json",
    ];

    let mut last_err = None;
    for url in &urls {
        match download_with_retry_bytes(url, 2, 120, "llm-benchmark-runner") {
            Ok(bytes) => {
                // Try direct parse
                if let Ok(items) = serde_json::from_slice::<Vec<HalUBenchItem>>(&bytes) {
                    let items: Vec<HalUBenchItem> = items.into_iter().take(max_items).collect();
                    fs::write(&path, serde_json::to_string_pretty(&items).unwrap())
                        .expect("Failed to save HaluBench");
                    return Ok(items);
                }
                // Try nested structure with common keys
                if let Ok(val) = serde_json::from_slice::<serde_json::Value>(&bytes) {
                    // Check for common top-level keys
                    for key in &["data", "test", "train", "samples", "items"] {
                        if let Some(arr) = val.get(key).and_then(|v| v.as_array()) {
                            let mut items = Vec::new();
                            for obj in arr.iter().take(max_items) {
                                if let Some(item) = parse_halubench_item(obj) {
                                    items.push(item);
                                }
                            }
                            if !items.is_empty() {
                                fs::write(&path, serde_json::to_string_pretty(&items).unwrap())
                                    .expect("Failed to save HaluBench");
                                return Ok(items);
                            }
                        }
                    }
                }
                eprintln!("  Failed to parse HaluBench from {}", url);
                last_err = Some(anyhow::anyhow!(
                    "Failed to parse HaluBench from {} — unexpected format",
                    url
                ));
            }
            Err(e) => {
                last_err = Some(anyhow::anyhow!("Failed to download from {}: {}", url, e));
            }
        }
    }

    let err = last_err.unwrap_or(anyhow::anyhow!("No download sources available"));
    eprintln!("  Error: {}", err);
    eprintln!(
        "  Please manually download the HaluBench dataset and place it at: {}",
        path.display()
    );
    Err(err)
}

fn parse_halubench_item(obj: &serde_json::Value) -> Option<HalUBenchItem> {
    let question = obj.get("question")?.as_str()?.to_string();
    let context = obj.get("context")?.as_str()?.to_string();
    let answer = obj.get("answer")?.as_str()?.to_string();
    let label = obj
        .get("label")
        .and_then(|l| l.as_str())
        .unwrap_or("true")
        .to_string();
    let dataset = obj
        .get("dataset")
        .and_then(|d| d.as_str())
        .unwrap_or("")
        .to_string();
    let task_id = obj
        .get("task_id")
        .and_then(|t| t.as_str())
        .unwrap_or("")
        .to_string();

    Some(HalUBenchItem {
        question,
        context,
        answer,
        label,
        dataset,
        task_id,
    })
}

/// Check if the model's answer contains the key information from the ground truth.
fn keyword_match(model_answer: &str, ground_truth: &str) -> bool {
    let model_lower = model_answer.to_lowercase();
    let gt_lower = ground_truth.to_lowercase();

    // Direct match
    if model_lower == gt_lower {
        return true;
    }

    // Ground truth contains model answer (model was more specific)
    if gt_lower.contains(&model_lower) && model_lower.len() > 3 {
        return true;
    }

    // Model answer contains ground truth (model was more verbose but correct)
    if model_lower.contains(&gt_lower) && gt_lower.len() > 3 {
        return true;
    }

    // Key word overlap — extract significant words from ground truth
    let gt_words: Vec<&str> = gt_lower
        .split_whitespace()
        .filter(|w| w.len() > 2)
        .collect();

    if !gt_words.is_empty() {
        let match_count = gt_words.iter().filter(|w| model_lower.contains(*w)).count();
        let match_rate = match_count as f64 / gt_words.len() as f64;
        if match_rate >= 0.5 && gt_words.len() <= 4 {
            return true;
        }
        if match_rate >= 0.7 {
            return true;
        }
    }

    false
}

impl Benchmark for HaluBenchBenchmark {
    fn name(&self) -> &str {
        "halubench"
    }

    fn display_name(&self) -> &'static str {
        "HaluBench (RAG Hallucination)"
    }

    fn category(&self) -> BenchmarkCategory {
        BenchmarkCategory::Hallucination
    }

    fn pre_execute(&self, config: &yaml_serde::Value) -> Result<()> {
        let max_items = config
            .get("max_items")
            .and_then(|v| v.as_u64())
            .unwrap_or(200) as usize;
        let items = load_halubench_dataset(max_items)?;
        println!(
            "  HaluBench: {} instances loaded (max: {})",
            items.len(),
            max_items
        );
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

        let system_prompt = "You are a question-answering assistant that must answer based ONLY on the provided context. If the context doesn't contain enough information to answer the question, respond with 'I cannot answer this from the given context.' Do not use outside knowledge.";

        let user_prompt = r#"Context: The Eiffel Tower is a wrought-iron lattice tower on the Champ de Mars in Paris, France. It was built from 1887 to 1889 as the entrance to the 1889 World's Fair.
Question: Where is the Eiffel Tower located?
Answer: Paris, France

Context: The Great Wall of China is a series of fortifications made of stone, brick, and other materials, generally built along an east-to-west line across the historical northern borders of China.
Question: What was the Great Wall built from?
Answer: Stone, brick, and other materials

Context: <context>{context}</context>
Question: <question>{question}</question>
Answer:"#;

        let prompt = user_prompt
            .replace("{context}", &fence_prompt_value(&item.context))
            .replace("{question}", &fence_prompt_value(&item.question));

        let response = tracker.chat_completion(&model.model_name, system_prompt, &prompt)?;
        let response_trimmed = response.trim();

        // Check for unanswerable detection
        let is_unanswerable = response_trimmed.to_lowercase().contains("cannot answer")
            || response_trimmed
                .to_lowercase()
                .contains("not enough information")
            || response_trimmed.to_lowercase().contains("cannot determine")
            || response_trimmed.to_lowercase().contains("i don't know")
            || response_trimmed.to_lowercase().contains("not provided");

        // Scoring
        let is_correct = match item.label.to_lowercase().as_str() {
            "true" | "1" => keyword_match(response_trimmed, &item.answer),
            "false" | "0" => {
                // For false/unanswerable, correct = model said it can't answer
                is_unanswerable
            }
            "unanswerable" | "not_enough_info" | "ne" => {
                // Model correctly identified unanswerability
                is_unanswerable
            }
            _ => keyword_match(response_trimmed, &item.answer),
        };

        Ok(Some(
            TaskResult::new(
                format!("task-{}", idx),
                is_correct,
                if is_correct { 1.0 } else { 0.0 },
                vec![item.label.clone()],
            )
            .with_metadata(Some(serde_json::json!({
                "question": item.question,
                "expected": item.answer,
                "label": item.label,
                "response": response_trimmed,
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
                            Score::float(rate * 100.0, ScoreUnit::Percent)
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
                    "HaluBench: {}/{} correct ({:.1}%)",
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
    fn test_keyword_match_exact() {
        assert!(keyword_match("Paris", "Paris"));
    }

    #[test]
    fn test_keyword_match_contains() {
        assert!(keyword_match("The answer is Paris, France", "Paris"));
    }

    #[test]
    fn test_keyword_match_no_match() {
        assert!(!keyword_match("London", "Paris"));
    }
}
