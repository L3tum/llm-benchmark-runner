use crate::benchmarks::Benchmark;
use crate::config::Model;
use crate::download::download_with_retry_bytes;
use crate::shared::{BenchmarkCategory, BenchmarkResult, Score, ScoreUnit, TaskResult};
use crate::token_tracker::TokenTracker;
use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::fs;
use std::sync::Mutex;

pub struct FeverBenchmark {
    state: Mutex<FeverState>,
}

#[derive(Clone, serde::Serialize, serde::Deserialize)]
struct FeverState {
    items: Vec<FeverItem>,
    current_idx: usize,
}

impl Default for FeverBenchmark {
    fn default() -> Self {
        Self {
            state: Mutex::new(FeverState {
                items: Vec::new(),
                current_idx: 0,
            }),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct FeverItem {
    claim: String,
    label: String, // "SUPPORTS", "REFUTES", "NOT ENOUGH INFO"
}

fn load_fever_dataset() -> Vec<FeverItem> {
    let cache_dir = dirs::cache_dir()
        .unwrap_or_default()
        .join("llm-benchmark-runner")
        .join("fever");
    let path = cache_dir.join("fever_dev.json");
    let url = "https://fever.ai/download/fever/shared_task_dev.jsonl";

    if path.exists() {
        let content = fs::read_to_string(&path).expect("Failed to read cached FEVER");
        return serde_json::from_str(&content).expect("Failed to parse FEVER");
    }

    fs::create_dir_all(&cache_dir).expect("Failed to create cache dir");
    println!("  Downloading FEVER dev dataset...");
    let bytes = download_with_retry_bytes(url, 3, 60, "llm-benchmark-runner")
        .expect("Failed to download FEVER");

    // shared_task_dev.jsonl: one claim per line {claim, verdict, evidence, id}.
    let content = String::from_utf8(bytes.to_vec()).expect("Failed to decode UTF-8");
    let items: Vec<FeverItem> = content
        .lines()
        .filter_map(|line| -> Option<FeverItem> {
            let v = serde_json::from_str::<serde_json::Value>(line).ok()?;
            let claim = v.get("claim")?.as_str()?.to_string();
            let label = v
                .get("verdict")
                .and_then(|l| l.as_str())
                .unwrap_or("NOT ENOUGH INFO")
                .to_string();
            Some(FeverItem { claim, label })
        })
        .collect();
    fs::write(
        &path,
        serde_json::to_vec(&items).expect("Failed to save FEVER"),
    )
    .expect("Failed to save FEVER");
    items
}

impl Benchmark for FeverBenchmark {
    fn name(&self) -> &str {
        "fever"
    }

    fn display_name(&self) -> &'static str {
        "FEVER (Fact Extraction and VERification)"
    }

    fn category(&self) -> BenchmarkCategory {
        BenchmarkCategory::Hallucination
    }

    fn pre_execute(&self, _config: &yaml_serde::Value) -> Result<()> {
        let items = load_fever_dataset();
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

        let system_prompt = "You are a fact verification assistant. Given a claim, determine whether it is SUPPORTS (the claim is true based on factual knowledge), REFUTES (the claim is false), or NOT ENOUGH INFO (you cannot determine its truth from your knowledge). Respond with only one of these three labels.";

        let user_prompt = r#"Claim: The Eiffel Tower is located in Paris.
Label: SUPPORTS

Claim: Water boils at 100 degrees Celsius at sea level.
Label: SUPPORTS

Claim: Albert Einstein was born in 1879.
Label: SUPPORTS

Claim: The moon is made of cheese.
Label: REFUTES

Claim: The Earth is flat.
Label: REFUTES

Claim: A certain unverified conspiracy theory about a famous person.
Label: NOT ENOUGH INFO

Claim: {claim}
Label:"#;

        let prompt = user_prompt.replace("{claim}", &item.claim);
        let response = tracker.chat_completion(&model.model_name, system_prompt, &prompt)?;

        let response = response.trim().to_uppercase();
        let is_correct = response.contains("SUPPORTS") && item.label == "SUPPORTS"
            || response.contains("REFUTES") && item.label == "REFUTES"
            || response.contains("NOT ENOUGH INFO") && item.label == "NOT ENOUGH INFO";

        let categories = vec![item.label.clone()];

        Ok(Some(
            TaskResult::new(
                format!("task-{}", idx),
                is_correct,
                if is_correct { 1.0 } else { 0.0 },
                categories,
            )
            .with_metadata(Some(serde_json::json!({
                "claim": item.claim,
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
                    "FEVER: {}/{} correct ({:.1}%)",
                    correct,
                    total,
                    accuracy * 100.0
                ),
            }],
            raw: raw.clone(),
        })
    }
}
