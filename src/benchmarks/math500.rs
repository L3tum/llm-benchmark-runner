use crate::benchmarks::Benchmark;
use crate::config::Model;
use crate::download::download_with_retry_bytes;
use crate::reports::model::{BenchmarkCategory, BenchmarkResult, TaskResult};
use crate::token_tracker::TokenTracker;
use anyhow::Result;
use regex::Regex;
use std::collections::{BTreeMap, HashMap};
use std::fs;
use std::path::PathBuf;
use std::sync::Mutex;

/// Single MATH-500 problem.
#[derive(Debug, Clone, serde::Deserialize)]
pub struct Math500Item {
    pub problem: String,
    pub answer: String,
    pub subject: String,
}

pub struct Math500Benchmark {
    state: Mutex<Math500State>,
}

struct Math500State {
    items: Vec<Math500Item>,
    current_idx: usize,
}

impl Default for Math500Benchmark {
    fn default() -> Self {
        Self {
            state: Mutex::new(Math500State {
                items: Vec::new(),
                current_idx: 0,
            }),
        }
    }
}

fn group_by_subject(items: Vec<Math500Item>) -> HashMap<String, Vec<Math500Item>> {
    let mut groups: HashMap<String, Vec<Math500Item>> = HashMap::new();
    for item in items {
        let subject = item.subject.clone();
        groups.entry(subject).or_default().push(item);
    }
    groups
}

impl Benchmark for Math500Benchmark {
    fn name(&self) -> &str {
        "math500"
    }

    fn display_name(&self) -> &'static str {
        "MATH-500"
    }

    fn category(&self) -> BenchmarkCategory {
        BenchmarkCategory::Math
    }

    fn pre_execute(&self, config: &yaml_serde::Value) -> Result<()> {
        let num_samples: Option<i64> = config.get("num_samples").and_then(|v| v.as_i64());
        let subjects_filter = config.get("subjects");
        let subjects: Option<Vec<String>> = match subjects_filter {
            Some(s) if s.is_string() => Some(
                s.as_str()
                    .unwrap()
                    .split(',')
                    .map(|s| s.trim().to_string())
                    .collect(),
            ),
            Some(s) if s.is_null() => None,
            _ => None,
        };

        let data_path = self.download_dataset()?;
        let content = fs::read_to_string(&data_path)?;
        let all_items: Vec<Math500Item> = serde_json::from_str(&content)?;
        let all_data = group_by_subject(all_items);

        let subjects_to_eval: Vec<String> = if let Some(subj) = &subjects {
            let mut result = Vec::new();
            for s in subj {
                if all_data.contains_key(s) {
                    result.push(s.clone());
                } else {
                    eprintln!(
                        "  WARNING: MATH-500 subject '{}' not found. Available: {:?}",
                        s,
                        all_data.keys()
                    );
                }
            }
            result
        } else {
            all_data.keys().cloned().collect()
        };

        // Flatten subjects into a single list
        let mut items = Vec::new();
        for subject in &subjects_to_eval {
            if let Some(questions) = all_data.get(subject) {
                let questions = match num_samples {
                    Some(n) if questions.len() > n as usize => questions[..n as usize].to_vec(),
                    _ => questions.clone(),
                };
                items.extend(questions);
            }
        }

        println!(
            "Evaluating MATH-500: {} total problems (zero-shot CoT)",
            items.len()
        );

        let mut state = self.state.lock().unwrap();
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
            let mut state = self.state.lock().unwrap();
            if state.current_idx >= state.items.len() {
                return Ok(None);
            }
            let idx = state.current_idx;
            let q = state.items[idx].clone();
            state.current_idx += 1;
            (q, idx)
        };

        let prompt = format!(
            "You are a math competition solver. Solve the following problem step by step. The answer should be put in the format of \"\\\\boxed{{answer}}\" at the end.\n\n{}\nPlease reason step by step, and put your final answer within \\\\boxed{{}}.",
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
                vec![q.subject.clone()],
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
        use crate::reports::model::{BreakdownTable, Score, ScoreUnit};

        let (total, correct, _wrong, output_tokens, thinking_tokens) = {
            if let Some(per_task) = raw.get("per_task").and_then(|v| v.as_array()) {
                let total = per_task.len() as i64;
                let correct = per_task
                    .iter()
                    .filter(|t| t.get("passed").and_then(|v| v.as_bool()).unwrap_or(false))
                    .count() as i64;
                let wrong = total - correct;
                let out: i64 = per_task
                    .iter()
                    .filter_map(|t| t.get("output_tokens").and_then(|v| v.as_i64()))
                    .sum();
                let think: i64 = per_task
                    .iter()
                    .filter_map(|t| t.get("thinking_tokens").and_then(|v| v.as_i64()))
                    .sum();
                (total, correct, wrong, out, think)
            } else {
                (
                    raw.get("total_questions")
                        .and_then(|v| v.as_i64())
                        .unwrap_or(0),
                    raw.get("correct").and_then(|v| v.as_i64()).unwrap_or(0),
                    raw.get("wrong").and_then(|v| v.as_i64()).unwrap_or(0),
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

        // Subject breakdown from per_task
        let mut subject_rows = BTreeMap::new();
        if let Some(per_task) = raw.get("per_task").and_then(|v| v.as_array()) {
            let mut cat_map: BTreeMap<String, (i64, i64)> = BTreeMap::new();
            for task in per_task {
                if let Some(cats) = task.get("categories").and_then(|v| v.as_array()) {
                    for cat in cats {
                        if let Some(cat_name) = cat.as_str() {
                            let entry = cat_map.entry(cat_name.to_string()).or_insert((0, 0));
                            entry.1 += 1;
                            if task
                                .get("passed")
                                .and_then(|v| v.as_bool())
                                .unwrap_or(false)
                            {
                                entry.0 += 1;
                            }
                        }
                    }
                }
            }
            for (subject, (corr, total)) in cat_map {
                let acc = if total > 0 {
                    corr as f64 / total as f64
                } else {
                    0.0
                };
                let mut row_scores = BTreeMap::new();
                row_scores.insert(
                    "accuracy".to_string(),
                    Score::float(acc * 100.0, ScoreUnit::Percent),
                );
                row_scores.insert(
                    "correct".to_string(),
                    Score::integer(corr, ScoreUnit::Count),
                );
                row_scores.insert(
                    "wrong".to_string(),
                    Score::integer(total - corr, ScoreUnit::Count),
                );
                subject_rows.insert(subject, row_scores);
            }
        }

        // Also try legacy format
        if subject_rows.is_empty() {
            if let Some(subjects) = raw.get("results_by_subject").and_then(|v| v.as_object()) {
                for (subject, data) in subjects {
                    if let Some(obj) = data.as_object() {
                        let acc = obj.get("acc").and_then(|v| v.as_f64()).unwrap_or(0.0);
                        let correct = obj.get("corr").and_then(|v| v.as_i64()).unwrap_or(0);
                        let wrong = obj.get("wrong").and_then(|v| v.as_i64()).unwrap_or(0);
                        let mut row_scores = BTreeMap::new();
                        row_scores.insert(
                            "accuracy".to_string(),
                            Score::float(acc, ScoreUnit::Percent),
                        );
                        row_scores.insert(
                            "correct".to_string(),
                            Score::integer(correct, ScoreUnit::Count),
                        );
                        row_scores
                            .insert("wrong".to_string(), Score::integer(wrong, ScoreUnit::Count));
                        subject_rows.insert(subject.clone(), row_scores);
                    }
                }
            }
        }

        let mut breakdowns = BTreeMap::new();
        if !subject_rows.is_empty() {
            breakdowns.insert(
                "subjects".to_string(),
                BreakdownTable {
                    title: "Subject Breakdown".to_string(),
                    rows: subject_rows,
                },
            );
        }

        Ok(BenchmarkResult {
            scores,
            breakdowns,
            error_classification: BTreeMap::new(),
            artifacts: vec![],
            diagnostics: vec![],
            raw: raw.clone(),
        })
    }
}

impl Math500Benchmark {
    pub fn download_dataset(&self) -> Result<PathBuf> {
        let cache_dir = dirs::cache_dir()
            .unwrap_or_default()
            .join("llm-benchmark-runner")
            .join("math500");
        fs::create_dir_all(&cache_dir)?;
        let path = cache_dir.join("MATH-500.json");
        if path.exists() {
            return Ok(path);
        }

        // Download from HuggingFaceH4/MATH-500
        let url =
            "https://huggingface.co/datasets/HuggingFaceH4/MATH-500/resolve/main/MATH-500.json";
        println!("  Downloading MATH-500 data...");
        let bytes = download_with_retry_bytes(url, 3, 60, "llm-benchmark-runner")?;
        fs::write(&path, bytes)?;
        Ok(path)
    }
}

/// Extract integer answer from boxed notation like \boxed{123}.
fn extract_int_answer(text: &str) -> Option<String> {
    let re = Regex::new(r"\\boxed\{(\d+)\}")
        .ok()
        .and_then(|r| r.captures_iter(text).last())
        .and_then(|caps| caps.get(1))
        .map(|m| m.as_str().to_string());
    re
}
