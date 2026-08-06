use crate::benchmarks::Benchmark;
use crate::config::Model;
use crate::download::download_with_retry_bytes;
use crate::shared::{
    fence_prompt_value, BenchmarkCategory, BenchmarkResult, Score, ScoreUnit, TaskResult,
};
use crate::token_tracker::TokenTracker;
use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::fs;
use std::sync::Mutex;

pub struct PopQABenchmark {
    state: Mutex<PopQAState>,
}

struct PopQAState {
    items: Vec<PopQAItem>,
    current_idx: usize,
}

impl Default for PopQABenchmark {
    fn default() -> Self {
        Self {
            state: Mutex::new(PopQAState {
                items: Vec::new(),
                current_idx: 0,
            }),
        }
    }
}

#[derive(Debug, Clone, Deserialize, Serialize)]
struct PopQAItem {
    question: String,
    subject_id: Option<String>,
    relation_id: Option<String>,
    #[serde(rename = "answer_argument")]
    answer_argument: Option<String>,
    #[serde(rename = "answer_argument_name")]
    answer_argument_name: Option<String>,
    // PopQA has many fields; we only need these
}

fn load_popqa_dataset(max_items: usize) -> Result<Vec<PopQAItem>> {
    let cache_dir = dirs::cache_dir()
        .unwrap_or_default()
        .join("llm-benchmark-runner")
        .join("popqa");
    let path = cache_dir.join("popqa.json");
    let url = "https://huggingface.co/datasets/akariasai/PopQA/resolve/main/test.tsv";

    if path.exists() {
        let content = fs::read_to_string(&path)?;
        let items: Vec<PopQAItem> = serde_json::from_str(&content)?;
        return Ok(items.into_iter().take(max_items).collect());
    }

    fs::create_dir_all(&cache_dir)?;
    println!(
        "  Downloading PopQA dataset (up to {} instances)...",
        max_items
    );

    // akariasai/PopQA ships a tab-separated test.tsv (header + rows).
    let bytes = download_with_retry_bytes(url, 3, 120, "llm-benchmark-runner")?;
    let content = String::from_utf8(bytes.to_vec())?;
    let mut items = Vec::new();
    for line in content.lines().skip(1) {
        if items.len() >= max_items {
            break;
        }
        if line.trim().is_empty() {
            continue;
        }
        let f: Vec<&str> = line.split('\t').collect();
        if f.len() < 17 {
            continue;
        }
        let obj = f[3].trim();
        if obj.is_empty() {
            continue;
        }
        items.push(PopQAItem {
            question: f[15].trim().to_string(),
            subject_id: Some(f[4].trim().to_string()),
            relation_id: Some(f[5].trim().to_string()),
            answer_argument: Some(f[6].trim().to_string()),
            answer_argument_name: Some(obj.to_string()),
        });
    }
    fs::write(&path, serde_json::to_string_pretty(&items)?).expect("Failed to save PopQA");
    Ok(items.into_iter().take(max_items).collect())
}

fn contains_entity(response: &str, entity: &str) -> bool {
    if entity.is_empty() {
        return false;
    }
    let response_lower = response.to_lowercase();
    let entity_lower = entity.to_lowercase();

    // Direct substring match
    if response_lower.contains(&entity_lower) {
        return true;
    }

    // Handle multi-word entities: check if all words appear in order
    let entity_words: Vec<&str> = entity_lower.split_whitespace().collect();
    if entity_words.len() > 1 {
        let mut pos = 0;
        let mut found = true;
        for word in &entity_words {
            if let Some(idx) = response_lower[pos..].find(*word) {
                pos += idx + word.len();
            } else {
                found = false;
                break;
            }
        }
        if found {
            return true;
        }
    }

    false
}

impl Benchmark for PopQABenchmark {
    fn name(&self) -> &str {
        "popqa"
    }

    fn display_name(&self) -> &'static str {
        "PopQA (Knowledge-based QA)"
    }

    fn category(&self) -> BenchmarkCategory {
        BenchmarkCategory::Knowledge
    }

    fn pre_execute(&self, config: &yaml_serde::Value) -> Result<()> {
        let max_items = config
            .get("max_items")
            .and_then(|v| v.as_u64())
            .unwrap_or(500) as usize;
        let items = load_popqa_dataset(max_items)?;
        println!(
            "  PopQA: {} instances loaded (max: {})",
            items.len(),
            max_items
        );
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

        // Use answer_argument_name if available (often more natural), otherwise answer_argument
        let gold_answer = item
            .answer_argument_name
            .clone()
            .or_else(|| item.answer_argument.clone())
            .unwrap_or_default();

        if gold_answer.is_empty() {
            // Skip items without answers
            return Ok(Some(
                TaskResult::new(
                    format!("task-{}", idx),
                    false,
                    0.0,
                    vec!["skipped".to_string()],
                )
                .with_metadata(Some(serde_json::json!({
                    "question": item.question,
                    "reason": "no answer available",
                }))),
            ));
        }

        let system_prompt =
            "You are a knowledgeable assistant. Answer each question with a single entity (person, place, organization, thing, etc.) or a short factual phrase. Be concise.";

        let user_prompt = r#"Question: Who directed the movie The Matrix?
Answer: The Wachowskis

Question: What is the capital of Japan?
Answer: Tokyo

Question: Who wrote Romeo and Juliet?
Answer: William Shakespeare

Question: What planet is known as the Red Planet?
Answer: Mars

Question: {question}
Answer:"#;

        let prompt = user_prompt.replace("{question}", &fence_prompt_value(&item.question));
        let response = tracker.chat_completion(&model.model_name, system_prompt, &prompt)?;

        let response_trimmed = response.trim();
        let is_correct = contains_entity(response_trimmed, &gold_answer);

        Ok(Some(
            TaskResult::new(
                format!("task-{}", idx),
                is_correct,
                if is_correct { 1.0 } else { 0.0 },
                vec![],
            )
            .with_metadata(Some(serde_json::json!({
                "question": item.question,
                "expected": gold_answer,
                "response": response_trimmed,
            }))),
        ))
    }

    fn to_report_result(&self, b: &BenchmarkResult) -> Result<BenchmarkResult> {
        let raw = &b.raw;

        let (total, correct, output_tokens, thinking_tokens, skipped) = {
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
                let skipped = per_task
                    .iter()
                    .filter(|t| {
                        t.get("categories")
                            .and_then(|v| v.as_array())
                            .and_then(|arr| arr.first())
                            .and_then(|v| v.as_str())
                            == Some("skipped")
                    })
                    .count() as i64;
                (total, correct, out, think, skipped)
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
                    0,
                )
            }
        };

        let effective_total = total.saturating_sub(skipped);
        let hit_rate = if effective_total > 0 {
            correct as f64 / effective_total as f64
        } else {
            0.0
        };

        let mut scores = BTreeMap::new();
        scores.insert(
            "hit_rate".to_string(),
            Score::float(hit_rate * 100.0, ScoreUnit::Percent)
                .primary(true)
                .higher_is_better(true)
                .display(format!("{:.1}% entity match rate", hit_rate * 100.0)),
        );
        scores.insert(
            "total".to_string(),
            Score::integer(effective_total, ScoreUnit::Count),
        );
        scores.insert(
            "correct".to_string(),
            Score::integer(correct, ScoreUnit::Count),
        );
        if skipped > 0 {
            scores.insert(
                "skipped".to_string(),
                Score::integer(skipped, ScoreUnit::Count)
                    .display(format!("{} skipped (no answer)", skipped)),
            );
        }
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
                    "PopQA: {} entity hits out of {} ({:.1}%)",
                    correct,
                    effective_total,
                    hit_rate * 100.0
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
    fn test_contains_entity_exact() {
        assert!(contains_entity("Paris", "Paris"));
    }

    #[test]
    fn test_contains_entity_substring() {
        assert!(contains_entity("The answer is Paris, France", "Paris"));
    }

    #[test]
    fn test_contains_entity_multiword_ordered() {
        assert!(contains_entity(
            "Barack Obama was the 44th president",
            "Barack Obama"
        ));
    }

    #[test]
    fn test_contains_entity_no_match() {
        assert!(!contains_entity("London", "Paris"));
    }
}
