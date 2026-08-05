use crate::benchmarks::Benchmark;
use crate::config::Model;
use crate::download::download_with_retry_bytes;
use crate::shared::{BenchmarkCategory, BenchmarkResult, Score, ScoreUnit, TaskResult};
use crate::token_tracker::TokenTracker;
use anyhow::{Context, Result};
use once_cell::sync::Lazy;
use regex::Regex;
use std::collections::BTreeMap;
use std::fs;
use std::path::PathBuf;
use std::sync::Mutex;

/// Single AIME problem with problem statement and integer answer.
#[derive(Debug, Clone, serde::Deserialize)]
pub struct AimeItem {
    pub problem: String,
    pub answer: String,
}

pub struct AimeBenchmark {
    state: Mutex<AimeState>,
}

struct AimeState {
    items: Vec<AimeItem>,
    current_idx: usize,
}

impl Default for AimeBenchmark {
    fn default() -> Self {
        Self {
            state: Mutex::new(AimeState {
                items: Vec::new(),
                current_idx: 0,
            }),
        }
    }
}

fn load_aime_json(path: &PathBuf) -> Result<Vec<AimeItem>> {
    let content = fs::read_to_string(path)?;
    // The JSON file contains an array of objects with `problem` and `answer` fields.
    // The `answer` can be an integer or a string; we'll accept either.
    let items: Vec<serde_json::Value> = serde_json::from_str(&content)?;
    let mut result = Vec::new();
    for item in items {
        let problem = item["problem"].as_str().unwrap_or("").to_string();
        let answer = if let Some(s) = item["answer"].as_str() {
            s.to_string()
        } else if let Some(n) = item["answer"].as_i64() {
            n.to_string()
        } else {
            item["answer"]
                .as_u64()
                .map(|n| n.to_string())
                .unwrap_or_default()
        };
        result.push(AimeItem { problem, answer });
    }
    Ok(result)
}

impl Benchmark for AimeBenchmark {
    fn name(&self) -> &str {
        "aime"
    }

    fn display_name(&self) -> &'static str {
        "AIME"
    }

    fn category(&self) -> BenchmarkCategory {
        BenchmarkCategory::Math
    }

    fn pre_execute(&self, config: &yaml_serde::Value) -> Result<()> {
        let year = config
            .get("year")
            .and_then(|v| v.as_str())
            .unwrap_or("2025");
        let data_path = self.download_dataset(year)?;
        let num_samples: Option<i64> = config.get("num_samples").and_then(|v| v.as_i64());
        let all_items = load_aime_json(&data_path)?;
        let items = match num_samples {
            Some(n) if all_items.len() > n as usize => all_items[..n as usize].to_vec(),
            _ => all_items,
        };

        println!(
            "Evaluating AIME {}: {} problems (zero-shot CoT)",
            year,
            items.len()
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
        let (q, idx) = {
            let mut state = self.state.lock().expect(crate::shared::MUTEX_PANIC_MSG);
            if state.current_idx >= state.items.len() {
                return Ok(None);
            }
            let idx = state.current_idx;
            let q = state.items[idx].clone();
            state.current_idx += 1;
            (q, idx)
        };

        let prompt = format!(
            "You are a math competition solver. Solve the following problem step by step. The answer is an integer between 000 and 999. Put your final answer in the format of \"\\boxed{{answer}}\" at the end.\n\n{}\nPlease reason step by step, and put your final answer within \\boxed{{}}.",
            q.problem
        );

        let response = tracker.chat_completion(&model.model_name, "", &prompt)?;
        let extracted_answer = extract_int_answer(&response);
        let is_correct = extracted_answer.as_deref() == Some(&q.answer);

        Ok(Some(
            TaskResult::new(
                format!("task-{}", idx),
                is_correct,
                if is_correct { 1.0 } else { 0.0 },
                vec![],
            )
            .with_metadata(Some(serde_json::json!({
                "expected": q.answer,
                "extracted": extracted_answer.unwrap_or_default(),
                "correct": is_correct,
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
                    raw.get("total_questions")
                        .and_then(|v| v.as_i64())
                        .unwrap_or(0),
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
        scores.insert(
            "correct".to_string(),
            Score::integer(correct, ScoreUnit::Count),
        );
        scores.insert(
            "total_questions".to_string(),
            Score::integer(total, ScoreUnit::Count),
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
            diagnostics: vec![],
            raw: raw.clone(),
        })
    }
}

impl AimeBenchmark {
    /// Download AIME dataset JSON from HuggingFace via the MathArena datasets viewer API.
    /// Supports `year` parameter to switch between AIME 2025 and 2026 datasets.
    pub fn download_dataset(&self, year: &str) -> Result<PathBuf> {
        let cache_dir = dirs::cache_dir()
            .unwrap_or_default()
            .join("llm-benchmark-runner")
            .join("aime");
        fs::create_dir_all(&cache_dir)?;
        let path = cache_dir.join(format!("aime_{}.json", year));
        if path.exists() {
            return Ok(path);
        }

        // Use the HuggingFace datasets viewer API to fetch all rows as JSON
        // MathArena/aime_2025 and MathArena/aime_2026 have train split with 30 rows
        let url = format!(
            "https://datasets-server.huggingface.co/rows?dataset=MathArena/aime_{0}&config=default&split=train&offset=0&length=100",
            year
        );
        println!(
            "  Downloading AIME {} data from MathArena via datasets viewer API...",
            year
        );

        let bytes = download_with_retry_bytes(&url, 3, 120, "llm-benchmark-runner")?;
        let api_result: serde_json::Value =
            serde_json::from_slice(&bytes).context("Failed to parse AIME API response")?;
        // Extract rows from the API response and save as a JSON array
        let rows: Vec<serde_json::Value> = api_result["rows"]
            .as_array()
            .ok_or_else(|| anyhow::anyhow!("Invalid API response"))?
            .iter()
            .map(|row| row["row"].clone())
            .collect();
        let json_bytes = serde_json::to_vec_pretty(&rows)?;
        fs::write(&path, json_bytes)?;
        Ok(path)
    }
}

static RE_BOXED: Lazy<Regex> = Lazy::new(|| Regex::new(r"\\boxed\{(\d+)\}").unwrap());

/// Extract a 3-digit integer answer from a model response using regex.
/// Looks for patterns like "\boxed{000}" or "\boxed{123}".
fn extract_int_answer(text: &str) -> Option<String> {
    RE_BOXED
        .captures_iter(text)
        .last()
        .and_then(|caps| caps.get(1))
        .map(|m| m.as_str().to_string())
}
