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

pub struct RaceBenchmark {
    state: Mutex<RaceState>,
}

#[derive(Clone, serde::Serialize, serde::Deserialize)]
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

#[derive(Debug, Clone, Serialize, Deserialize)]
struct RaceItem {
    passage: String,
    question: String,
    option_a: String,
    option_b: String,
    option_c: String,
    option_d: String,
    answer: String,
}

fn load_race_dataset() -> Result<Vec<RaceItem>> {
    use anyhow::Context;
    let cache_dir = dirs::cache_dir()
        .unwrap_or_default()
        .join("llm-benchmark-runner")
        .join("race");
    let path = cache_dir.join("race.json");
    let url =
        "https://huggingface.co/datasets/ehovy/race/resolve/main/all/test-00000-of-00001.parquet";

    if path.exists() {
        let content = fs::read_to_string(&path)?;
        return serde_json::from_str(&content).context("parse cached RACE");
    }

    fs::create_dir_all(&cache_dir).context("create race cache dir")?;
    println!("  Downloading RACE dataset...");
    let rows = download_parquet_records(url, 3, 60, "llm-benchmark-runner")
        .context("download RACE parquet")?;
    let items: Vec<RaceItem> = rows
        .iter()
        .map(|r| {
            let opts: Vec<String> = r
                .get("options")
                .and_then(|v| v.as_array())
                .map(|a| {
                    a.iter()
                        .filter_map(|x| x.as_str().map(String::from))
                        .collect()
                })
                .unwrap_or_default();
            RaceItem {
                passage: r
                    .get("article")
                    .and_then(|v| v.as_str())
                    .unwrap_or("")
                    .to_string(),
                question: r
                    .get("question")
                    .and_then(|v| v.as_str())
                    .unwrap_or("")
                    .to_string(),
                option_a: opts.first().cloned().unwrap_or_default(),
                option_b: opts.get(1).cloned().unwrap_or_default(),
                option_c: opts.get(2).cloned().unwrap_or_default(),
                option_d: opts.get(3).cloned().unwrap_or_default(),
                answer: r
                    .get("answer")
                    .and_then(|v| v.as_str())
                    .unwrap_or("")
                    .to_string(),
            }
        })
        .collect();
    fs::write(&path, serde_json::to_vec(&items)?).context("save RACE cache")?;
    Ok(items)
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
        let items = load_race_dataset()?;
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
