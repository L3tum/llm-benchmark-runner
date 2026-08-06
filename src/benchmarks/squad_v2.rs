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

pub struct SquadV2Benchmark {
    state: Mutex<SquadV2State>,
}

#[derive(Clone, serde::Serialize, serde::Deserialize)]
struct SquadV2State {
    items: Vec<SquadV2Item>,
    current_idx: usize,
}

impl Default for SquadV2Benchmark {
    fn default() -> Self {
        Self {
            state: Mutex::new(SquadV2State {
                items: Vec::new(),
                current_idx: 0,
            }),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[allow(dead_code)] // title field kept for schema alignment
struct SquadV2Item {
    title: String,
    context: String,
    question: String,
    answers: Option<Vec<SquadV2Answer>>,
    is_impossible: Option<bool>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[allow(dead_code)] // answer_start kept for schema alignment
struct SquadV2Answer {
    text: String,
    answer_start: i64,
}

fn load_squad_v2() -> Result<Vec<SquadV2Item>> {
    use anyhow::Context;
    let cache_dir = dirs::cache_dir()
        .unwrap_or_default()
        .join("llm-benchmark-runner")
        .join("squad_v2");
    let path = cache_dir.join("SQuAD2.0.json");
    let url =
        "https://huggingface.co/datasets/rajpurkar/squad_v2/resolve/main/squad_v2/validation-00000-of-00001.parquet";

    if path.exists() {
        let content = fs::read_to_string(&path)?;
        return serde_json::from_str(&content).context("parse cached SQuAD 2.0");
    }

    fs::create_dir_all(&cache_dir).context("create squad cache dir")?;
    println!("  Downloading SQuAD 2.0 dataset...");
    let rows = download_parquet_records(url, 3, 60, "llm-benchmark-runner")
        .context("download SQuAD parquet")?;
    let items: Vec<SquadV2Item> = rows
        .iter()
        .map(|r| {
            let answers: Vec<SquadV2Answer> = r
                .get("answers")
                .and_then(|v| v.as_array())
                .map(|a| {
                    a.iter()
                        .map(|ans| SquadV2Answer {
                            text: ans
                                .get("text")
                                .and_then(|t| t.as_str())
                                .unwrap_or("")
                                .to_string(),
                            answer_start: ans
                                .get("answer_start")
                                .and_then(|t| t.as_i64())
                                .unwrap_or(0),
                        })
                        .collect()
                })
                .unwrap_or_default();
            let is_impossible = answers.is_empty();
            SquadV2Item {
                title: r
                    .get("title")
                    .and_then(|v| v.as_str())
                    .unwrap_or("")
                    .to_string(),
                context: r
                    .get("context")
                    .and_then(|v| v.as_str())
                    .unwrap_or("")
                    .to_string(),
                question: r
                    .get("question")
                    .and_then(|v| v.as_str())
                    .unwrap_or("")
                    .to_string(),
                answers: Some(answers),
                is_impossible: Some(is_impossible),
            }
        })
        .collect();
    fs::write(&path, serde_json::to_vec(&items)?).context("save SQuAD cache")?;
    Ok(items)
}

impl Benchmark for SquadV2Benchmark {
    fn name(&self) -> &str {
        "squad_v2"
    }

    fn display_name(&self) -> &'static str {
        "SQuAD 2.0"
    }

    fn category(&self) -> BenchmarkCategory {
        BenchmarkCategory::Hallucination
    }

    fn pre_execute(&self, _config: &yaml_serde::Value) -> Result<()> {
        let items = load_squad_v2()?;
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

        let system_prompt =
            "You are a reading comprehension assistant. Answer the question using ONLY the provided context. If the answer cannot be found in the context, respond with 'no answer'.";

        let user_prompt = "Context: {context}\nQuestion: {question}\nAnswer:";
        let prompt = user_prompt
            .replace("{context}", &item.context)
            .replace("{question}", &item.question);

        let response = tracker.chat_completion(&model.model_name, system_prompt, &prompt)?;
        let response = response.trim();
        let is_unanswerable = item.is_impossible.unwrap_or(false);

        let (is_correct, f1, category) = if is_unanswerable {
            let is_correct = response.to_lowercase().contains("no answer")
                || response.to_lowercase().contains("not answerable")
                || response.to_lowercase().contains("cannot be found");
            (is_correct, 0.0, "unanswerable")
        } else if let Some(ref answers) = item.answers {
            if answers.is_empty() {
                (false, 0.0, "answerable")
            } else {
                let response_lower = response.trim().to_lowercase();
                let best_em = answers
                    .iter()
                    .any(|a| a.text.trim().to_lowercase() == response_lower);

                let best_f1 = answers
                    .iter()
                    .map(|a| compute_f1(&a.text, response))
                    .fold(0.0f64, f64::max);
                (best_em, best_f1, "answerable")
            }
        } else {
            (false, 0.0, "answerable")
        };

        Ok(Some(
            TaskResult::new(
                format!("task-{}", idx),
                is_correct,
                f1 * 100.0,
                vec![category.to_string()],
            )
            .with_metadata(Some(serde_json::json!({
                "is_unanswerable": is_unanswerable,
                "response": response,
                "f1": f1,
            }))),
        ))
    }

    fn to_report_result(&self, b: &BenchmarkResult) -> Result<BenchmarkResult> {
        let raw = &b.raw;

        let (
            total,
            em_correct,
            f1_total,
            unanswerable_total,
            unanswerable_correct,
            output_tokens,
            thinking_tokens,
        ) = {
            if let Some(per_task) = raw.get("per_task").and_then(|v| v.as_array()) {
                let total = per_task.len() as i64;
                let em_correct = per_task
                    .iter()
                    .filter(|t| {
                        t.get("passed").and_then(|v| v.as_bool()).unwrap_or(false)
                            && !t
                                .get("categories")
                                .and_then(|v| v.as_array())
                                .and_then(|arr| arr.first())
                                .and_then(|v| v.as_str())
                                .map(|s| s == "unanswerable")
                                .unwrap_or(false)
                    })
                    .count() as i64;
                let f1_total: f64 = per_task
                    .iter()
                    .filter(|t| {
                        !t.get("categories")
                            .and_then(|v| v.as_array())
                            .and_then(|arr| arr.first())
                            .and_then(|v| v.as_str())
                            .map(|s| s == "unanswerable")
                            .unwrap_or(false)
                    })
                    .filter_map(|t| t.get("score").and_then(|v| v.as_f64()))
                    .sum();
                let unanswerable_tasks: Vec<_> = per_task
                    .iter()
                    .filter(|t| {
                        t.get("categories")
                            .and_then(|v| v.as_array())
                            .and_then(|arr| arr.first())
                            .and_then(|v| v.as_str())
                            .map(|s| s == "unanswerable")
                            .unwrap_or(false)
                    })
                    .collect();
                let unanswerable_total = unanswerable_tasks.len() as i64;
                let unanswerable_correct = unanswerable_tasks
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
                (
                    total,
                    em_correct,
                    f1_total,
                    unanswerable_total,
                    unanswerable_correct,
                    out,
                    think,
                )
            } else {
                (
                    raw.get("total").and_then(|v| v.as_i64()).unwrap_or(0),
                    raw.get("exact_match_correct")
                        .and_then(|v| v.as_i64())
                        .unwrap_or(0),
                    raw.get("f1").and_then(|v| v.as_f64()).unwrap_or(0.0),
                    raw.get("unanswerable_total")
                        .and_then(|v| v.as_i64())
                        .unwrap_or(0),
                    raw.get("unanswerable_correct")
                        .and_then(|v| v.as_i64())
                        .unwrap_or(0),
                    raw.get("output_tokens")
                        .and_then(|v| v.as_i64())
                        .unwrap_or(0),
                    raw.get("thinking_tokens")
                        .and_then(|v| v.as_i64())
                        .unwrap_or(0),
                )
            }
        };

        let answerable_total = total.saturating_sub(unanswerable_total);
        let em_score = if answerable_total > 0 {
            em_correct as f64 / answerable_total as f64
        } else {
            0.0
        };
        let f1_score = if answerable_total > 0 {
            f1_total / answerable_total as f64
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
        scores.insert(
            "f1".to_string(),
            Score::float(f1_score * 100.0, ScoreUnit::Percent),
        );
        scores.insert(
            "unanswerable_accuracy".to_string(),
            Score::float(
                if unanswerable_total > 0 {
                    unanswerable_correct as f64 / unanswerable_total as f64 * 100.0
                } else {
                    0.0
                },
                ScoreUnit::Percent,
            ),
        );
        scores.insert("total".to_string(), Score::integer(total, ScoreUnit::Count));
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
                    "SQuAD 2.0: EM {:.1}%, F1 {:.1}% (answerable), unanswerable accuracy {:.1}% ({}/{})",
                    em_score * 100.0, f1_score * 100.0, unanswerable_correct as f64 / unanswerable_total.max(1) as f64 * 100.0,
                    unanswerable_correct, unanswerable_total
                ),
            }],
            raw: raw.clone(),
        })
    }
}

fn compute_f1(reference: &str, response: &str) -> f64 {
    let ref_lower = reference.to_lowercase();
    let resp_lower = response.to_lowercase();
    let reference_tokens: Vec<&str> = ref_lower.split_whitespace().collect();
    let response_tokens: Vec<&str> = resp_lower.split_whitespace().collect();

    let intersection: usize = reference_tokens
        .iter()
        .filter(|r| response_tokens.contains(r))
        .count();

    let precision = if response_tokens.is_empty() {
        0.0
    } else {
        intersection as f64 / response_tokens.len() as f64
    };
    let recall = if reference_tokens.is_empty() {
        0.0
    } else {
        intersection as f64 / reference_tokens.len() as f64
    };

    if precision + recall == 0.0 {
        0.0
    } else {
        2.0 * precision * recall / (precision + recall)
    }
}
