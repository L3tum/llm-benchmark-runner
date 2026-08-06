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

pub struct SciFactBenchmark {
    state: Mutex<SciFactState>,
}

struct SciFactState {
    items: Vec<SciFactItem>,
    current_idx: usize,
}

impl Default for SciFactBenchmark {
    fn default() -> Self {
        Self {
            state: Mutex::new(SciFactState {
                items: Vec::new(),
                current_idx: 0,
            }),
        }
    }
}

#[derive(Debug, Clone, Deserialize, Serialize)]
struct SciFactItem {
    claim: String,
    label: String, // "SUPPORTS" or "REFUTES"
    abstract_id: Option<String>,
    paper_title: Option<String>,
}

/// Wrapper for the SciFact dataset which has a nested structure.
#[derive(Debug, Deserialize)]
struct SciFactRaw {
    claims: Vec<SciFactClaimRaw>,
}

#[derive(Debug, Deserialize)]
struct SciFactClaimRaw {
    claim: String,
    label: String,
    abstract_id: Option<String>,
    paper_title: Option<String>,
    // Other fields we don't need
    #[allow(dead_code)]
    #[serde(flatten)]
    extra: serde_json::Map<String, serde_json::Value>,
}

fn load_scifact_dataset() -> Result<Vec<SciFactItem>> {
    let cache_dir = dirs::cache_dir()
        .unwrap_or_default()
        .join("llm-benchmark-runner")
        .join("scifact");
    let path = cache_dir.join("scifact.json");

    if path.exists() {
        let content = fs::read_to_string(&path).expect("Failed to read cached SciFact");
        return serde_json::from_str(&content).context("Failed to parse SciFact");
    }

    fs::create_dir_all(&cache_dir).expect("Failed to create cache dir");
    println!("  Downloading SciFact dataset...");

    // Try multiple sources
    let urls = [
        "https://huggingface.co/datasets/allenai/scifact/resolve/main/scifact.json",
        "https://raw.githubusercontent.com/allenai/scifact/master/data/scifact.json",
    ];

    let mut last_err = None;
    for url in &urls {
        match download_with_retry_bytes(url, 2, 60, "llm-benchmark-runner") {
            Ok(bytes) => {
                // Try direct parse as Vec<SciFactItem>
                if let Ok(items) = serde_json::from_slice::<Vec<SciFactItem>>(&bytes) {
                    fs::write(&path, &bytes).expect("Failed to save SciFact");
                    return Ok(items);
                }
                // Try nested structure
                if let Ok(raw) = serde_json::from_slice::<SciFactRaw>(&bytes) {
                    let items: Vec<SciFactItem> = raw
                        .claims
                        .into_iter()
                        .map(|c| SciFactItem {
                            claim: c.claim,
                            label: c.label,
                            abstract_id: c.abstract_id,
                            paper_title: c.paper_title,
                        })
                        .collect();
                    fs::write(&path, serde_json::to_string(&items).unwrap())
                        .expect("Failed to save SciFact");
                    return Ok(items);
                }
                // Try top-level JSON with a "claims" key
                if let Ok(val) = serde_json::from_slice::<serde_json::Value>(&bytes) {
                    if let Some(claims) = val.get("claims").and_then(|c| c.as_array()) {
                        let mut items = Vec::new();
                        for claim_obj in claims {
                            items.push(SciFactItem {
                                claim: claim_obj
                                    .get("claim")
                                    .and_then(|c| c.as_str())
                                    .unwrap_or("")
                                    .to_string(),
                                label: claim_obj
                                    .get("label")
                                    .and_then(|l| l.as_str())
                                    .unwrap_or("REFUTES")
                                    .to_string(),
                                abstract_id: claim_obj
                                    .get("abstract_id")
                                    .and_then(|a| a.as_str())
                                    .map(|s| s.to_string()),
                                paper_title: claim_obj
                                    .get("paper_title")
                                    .and_then(|p| p.as_str())
                                    .map(|s| s.to_string()),
                            });
                        }
                        fs::write(&path, serde_json::to_string(&items).unwrap())
                            .expect("Failed to save SciFact");
                        return Ok(items);
                    }
                }
                eprintln!("  Failed to parse SciFact from {}", url);
                last_err = Some(anyhow::anyhow!("Failed to parse SciFact from {}", url));
            }
            Err(e) => {
                last_err = Some(anyhow::anyhow!("Failed to download from {}: {}", url, e));
            }
        }
    }

    let err = last_err.unwrap_or(anyhow::anyhow!("No download sources available"));
    eprintln!("  Error: {}", err);
    eprintln!(
        "  Please manually download the SciFact dataset and place it at: {}",
        path.display()
    );
    Err(err)
}

impl Benchmark for SciFactBenchmark {
    fn name(&self) -> &str {
        "scifact"
    }

    fn display_name(&self) -> &'static str {
        "SciFact (Scientific Fact Verification)"
    }

    fn category(&self) -> BenchmarkCategory {
        BenchmarkCategory::Research
    }

    fn pre_execute(&self, _config: &yaml_serde::Value) -> Result<()> {
        let items = load_scifact_dataset()?;
        println!("  SciFact: {} claims loaded", items.len());
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

        let system_prompt = "You are a scientific fact verification assistant. Given a scientific claim, determine whether the claim is SUPPORTED or REFUTED based on your knowledge of scientific evidence. Respond with only SUPPORTS or REFUTES.";

        let user_prompt = r#"Claim: Exposure to air pollution increases the risk of cardiovascular disease.
Label: SUPPORTS

Claim: Vaccines cause autism.
Label: REFUTES

Claim: Exercise has no effect on mental health outcomes.
Label: REFUTES

Claim: Climate change is primarily caused by human activities.
Label: SUPPORTS

Claim: A certain unverified scientific claim about quantum computing.
Label: REFUTES

Claim: {claim}
Label:"#;

        let prompt = user_prompt.replace("{claim}", &fence_prompt_value(&item.claim));
        let response = tracker.chat_completion(&model.model_name, system_prompt, &prompt)?;

        let response_upper = response.trim().to_uppercase();
        let is_correct = match item.label.to_uppercase().as_str() {
            "SUPPORTS" => response_upper.contains("SUPPORTS"),
            "REFUTES" => response_upper.contains("REFUTES"),
            _ => false,
        };

        Ok(Some(
            TaskResult::new(
                format!("task-{}", idx),
                is_correct,
                if is_correct { 1.0 } else { 0.0 },
                vec![item.label.clone()],
            )
            .with_metadata(Some(serde_json::json!({
                "claim": item.claim,
                "expected": item.label,
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
                    "SciFact: {}/{} correct ({:.1}%)",
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
    fn test_scifact_default_state() {
        let bench = SciFactBenchmark::default();
        let state = bench.state.lock().unwrap();
        assert!(state.items.is_empty());
        assert_eq!(state.current_idx, 0);
    }
}
