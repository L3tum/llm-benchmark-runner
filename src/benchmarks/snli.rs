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

pub struct SnliBenchmark {
    state: Mutex<SnliState>,
}

struct SnliState {
    items: Vec<SnliItem>,
    current_idx: usize,
}

impl Default for SnliBenchmark {
    fn default() -> Self {
        Self {
            state: Mutex::new(SnliState {
                items: Vec::new(),
                current_idx: 0,
            }),
        }
    }
}

#[derive(Debug, Clone, Deserialize, Serialize)]
struct SnliItem {
    sentence1: String,  // premise
    sentence2: String,  // hypothesis
    gold_label: String, // "entailment", "contradiction", "neutral"
}

fn load_snli_dataset(max_items: usize) -> Result<Vec<SnliItem>> {
    let cache_dir = dirs::cache_dir()
        .unwrap_or_default()
        .join("llm-benchmark-runner")
        .join("snli");
    let path = cache_dir.join("snli_test.json");

    if path.exists() {
        let content = fs::read_to_string(&path).expect("Failed to read cached SNLI");
        let items: Vec<SnliItem> =
            serde_json::from_str(&content).context("Failed to parse SNLI")?;
        return Ok(items.into_iter().take(max_items).collect());
    }

    fs::create_dir_all(&cache_dir).expect("Failed to create cache dir");
    println!(
        "  Downloading SNLI dataset (test split, up to {} instances)...",
        max_items
    );

    // SNLI on HuggingFace — use the test split
    let url = "https://huggingface.co/datasets/stanfordnlp/snli/resolve/main/snli_1.0_test.jsonl";
    match download_with_retry_bytes(url, 3, 120, "llm-benchmark-runner") {
        Ok(bytes) => {
            let content = String::from_utf8(bytes.to_vec()).expect("Failed to decode UTF-8");
            let mut items = Vec::new();
            for line in content.lines() {
                if items.len() >= max_items || line.is_empty() {
                    continue;
                }
                if let Ok(val) = serde_json::from_str::<serde_json::Value>(line) {
                    let sentence1 = val
                        .get("sentence1")
                        .and_then(|s| s.as_str())
                        .unwrap_or("")
                        .to_string();
                    let sentence2 = val
                        .get("sentence2")
                        .and_then(|s| s.as_str())
                        .unwrap_or("")
                        .to_string();
                    let gold_label = match val.get("gold_label").and_then(|l| l.as_str()) {
                        Some(label) if label != "-" => label.to_string(),
                        _ => continue,
                    };
                    items.push(SnliItem {
                        sentence1,
                        sentence2,
                        gold_label,
                    });
                }
            }
            fs::write(&path, serde_json::to_string_pretty(&items).unwrap())
                .expect("Failed to save SNLI");
            return Ok(items);
        }
        Err(e) => {
            eprintln!("  Failed to download SNLI: {}", e);
        }
    }

    // Fallback: try alternate source
    let url2 = "https://raw.githubusercontent.com/salesforce/SNLI/master/snli_1.0_test.jsonl";
    match download_with_retry_bytes(url2, 2, 120, "llm-benchmark-runner") {
        Ok(bytes) => {
            let content = String::from_utf8(bytes.to_vec()).expect("Failed to decode UTF-8");
            let mut items = Vec::new();
            for line in content.lines() {
                if items.len() >= max_items || line.is_empty() {
                    continue;
                }
                if let Ok(val) = serde_json::from_str::<serde_json::Value>(line) {
                    let sentence1 = val
                        .get("sentence1")
                        .and_then(|s| s.as_str())
                        .unwrap_or("")
                        .to_string();
                    let sentence2 = val
                        .get("sentence2")
                        .and_then(|s| s.as_str())
                        .unwrap_or("")
                        .to_string();
                    let gold_label = match val.get("gold_label").and_then(|l| l.as_str()) {
                        Some(label) if label != "-" => label.to_string(),
                        _ => continue,
                    };
                    items.push(SnliItem {
                        sentence1,
                        sentence2,
                        gold_label,
                    });
                }
            }
            fs::write(&path, serde_json::to_string_pretty(&items).unwrap())
                .expect("Failed to save SNLI");
            return Ok(items);
        }
        Err(e) => {
            eprintln!("  Failed to download SNLI from fallback: {}", e);
        }
    }

    Err(anyhow::anyhow!(
        "Could not download SNLI dataset. Please manually download and place at: {}",
        path.display()
    ))
}

impl Benchmark for SnliBenchmark {
    fn name(&self) -> &str {
        "snli"
    }

    fn display_name(&self) -> &'static str {
        "SNLI (Natural Language Inference)"
    }

    fn category(&self) -> BenchmarkCategory {
        BenchmarkCategory::Reasoning
    }

    fn pre_execute(&self, config: &yaml_serde::Value) -> Result<()> {
        let max_items = config
            .get("max_items")
            .and_then(|v| v.as_u64())
            .unwrap_or(500) as usize;
        let items = load_snli_dataset(max_items)?;
        println!(
            "  SNLI: {} instances loaded (max: {})",
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

        let system_prompt = "You are a natural language inference assistant. Given a premise and a hypothesis, determine the relationship: ENTAILMENT (the hypothesis follows from the premise), CONTRADICTION (the hypothesis contradicts the premise), or NEUTRAL (the hypothesis is neither entailed nor contradicted). Respond with only ENTAILMENT, CONTRADICTION, or NEUTRAL.";

        let user_prompt = r#"Premise: A person is riding a bicycle on a road.
Hypothesis: Someone is on a bike.
Relationship: ENTAILMENT

Premise: A person is riding a bicycle on a road.
Hypothesis: A person is driving a car.
Relationship: CONTRADICTION

Premise: A person is riding a bicycle on a road.
Hypothesis: The road is made of asphalt.
Relationship: NEUTRAL

Premise: Two dogs are playing in the park.
Hypothesis: Animals are outdoors.
Relationship: ENTAILMENT

Premise: Two dogs are playing in the park.
Hypothesis: The dogs are sleeping.
Relationship: CONTRADICTION

Premise: A woman is reading a book in the library.
Hypothesis: A person is indoors.
Relationship: NEUTRAL

Premise: {premise}
Hypothesis: {hypothesis}
Relationship:"#;

        let prompt = user_prompt
            .replace("{premise}", &fence_prompt_value(&item.sentence1))
            .replace("{hypothesis}", &fence_prompt_value(&item.sentence2));

        let response = tracker.chat_completion(&model.model_name, system_prompt, &prompt)?;

        let response_upper = response.trim().to_uppercase();
        let predicted = if response_upper.contains("ENTAILMENT") {
            "entailment"
        } else if response_upper.contains("CONTRADICTION") {
            "contradiction"
        } else if response_upper.contains("NEUTRAL") {
            "neutral"
        } else {
            "unknown"
        };

        let is_correct = predicted == item.gold_label;

        Ok(Some(
            TaskResult::new(
                format!("task-{}", idx),
                is_correct,
                if is_correct { 1.0 } else { 0.0 },
                vec![item.gold_label.clone()],
            )
            .with_metadata(Some(serde_json::json!({
                "premise": item.sentence1,
                "hypothesis": item.sentence2,
                "expected": item.gold_label,
                "predicted": predicted,
                "response": response.trim(),
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
                    title: "Accuracy by NLI Label".to_string(),
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
                    "SNLI: {}/{} correct ({:.1}%)",
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
    fn test_snli_default_state() {
        let bench = SnliBenchmark::default();
        let state = bench.state.lock().unwrap();
        assert!(state.items.is_empty());
        assert_eq!(state.current_idx, 0);
    }
}
