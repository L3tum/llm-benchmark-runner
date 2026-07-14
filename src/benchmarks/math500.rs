use crate::client::Client;
use crate::config::Model;
use crate::download::download_with_retry_bytes;
use crate::reports::model::BenchmarkResult;
use anyhow::Result;
use regex::Regex;
use std::collections::{BTreeMap, HashMap};
use std::fs;
use std::path::PathBuf;

/// Single MATH-500 problem.
#[derive(Debug, Clone, serde::Deserialize)]
pub struct Math500Item {
    pub problem: String,
    pub answer: String,
    pub subject: String,
}

pub struct Math500Benchmark;

fn group_by_subject(items: Vec<Math500Item>) -> HashMap<String, Vec<Math500Item>> {
    let mut groups: HashMap<String, Vec<Math500Item>> = HashMap::new();
    for item in items {
        let subject = item.subject.clone();
        groups.entry(subject).or_default().push(item);
    }
    groups
}

impl super::Benchmark for Math500Benchmark {
    fn name(&self) -> &str {
        "math500"
    }

    fn display_name(&self) -> &'static str {
        "MATH-500"
    }

    fn category(&self) -> crate::reports::model::BenchmarkCategory {
        crate::reports::model::BenchmarkCategory::Math
    }

    fn pre_execute(&self, _config: &yaml_serde::Value) -> Result<()> {
        self.download_dataset()?;
        Ok(())
    }

    fn to_report_result(&self, raw: &serde_json::Value) -> Result<BenchmarkResult> {
        use crate::reports::model::{BreakdownTable, Score, ScoreUnit};

        let accuracy = raw.get("accuracy").and_then(|v| v.as_f64()).unwrap_or(0.0);
        let total_questions = raw
            .get("total_questions")
            .and_then(|v| v.as_i64())
            .unwrap_or(0);
        let output_tokens = raw
            .get("output_tokens")
            .and_then(|v| v.as_i64())
            .unwrap_or(0);
        let thinking_tokens = raw
            .get("thinking_tokens")
            .and_then(|v| v.as_i64())
            .unwrap_or(0);

        let mut scores = BTreeMap::new();
        scores.insert(
            "accuracy".to_string(),
            Score::float(accuracy, ScoreUnit::Percent)
                .primary(true)
                .higher_is_better(true),
        );
        scores.insert(
            "total_questions".to_string(),
            Score::integer(total_questions, ScoreUnit::Count),
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

        // Subject breakdown
        let mut subject_rows = BTreeMap::new();
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
                    row_scores.insert("wrong".to_string(), Score::integer(wrong, ScoreUnit::Count));
                    subject_rows.insert(subject.clone(), row_scores);
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
            artifacts: vec![],
            diagnostics: vec![],
            raw: raw.clone(),
        })
    }

    fn execute(&self, model: &Model, config: &yaml_serde::Value) -> Result<serde_json::Value> {
        let client = Client::new_with_model_params(&model.proxy, model.set_params.as_ref())?;
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

        let subjects_to_eval: Vec<String> = if let Some(subj) = subjects {
            let mut result = Vec::new();
            for s in &subj {
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

        let mut category_record: HashMap<String, serde_json::Value> = HashMap::new();
        let mut total_questions = 0usize;
        let mut total_output_tokens: u64 = 0;
        let mut total_thinking_tokens: u64 = 0;

        for subject in &subjects_to_eval {
            let questions = all_data
                .get(subject)
                .ok_or_else(|| anyhow::anyhow!("Subject {} not found", subject))?
                .clone();
            let questions = match num_samples {
                Some(n) if questions.len() > n as usize => questions[..n as usize].to_vec(),
                _ => questions,
            };

            println!(
                "\nEvaluating MATH-500 {}: {} problems (zero-shot CoT)",
                subject,
                questions.len()
            );

            let mut category_correct = 0usize;
            let mut category_total = 0usize;

            for q in &questions {
                let problem_text = q.problem.clone();
                let prompt = format!(
                    "You are a math competition solver. Solve the following problem step by step. The answer should be put in the format of \"\\boxed{{answer}}\" at the end.\n\n{}\nPlease reason step by step, and put your final answer within \\boxed{{}}.",
                    problem_text
                );

                let (response, output_tokens, thinking_tokens) =
                    client.chat_completion(&model.model_name, "", &prompt)?;
                total_output_tokens += output_tokens.unwrap_or(0);
                total_thinking_tokens += thinking_tokens.unwrap_or(0);
                let extracted_answer = extract_int_answer(&response);
                let is_correct = extracted_answer.as_deref() == Some(&q.answer);
                if is_correct {
                    category_correct += 1;
                }
                category_total += 1;
                total_questions += 1;
            }

            let accuracy = if category_total > 0 {
                category_correct as f64 / category_total as f64
            } else {
                0.0
            };
            let mut record = serde_json::Map::new();
            record.insert("acc".to_string(), serde_json::json!(accuracy));
            record.insert("corr".to_string(), serde_json::json!(category_correct));
            record.insert(
                "wrong".to_string(),
                serde_json::json!(category_total - category_correct),
            );
            category_record.insert(subject.clone(), serde_json::Value::Object(record));
        }

        let total_correct: i64 = category_record
            .values()
            .map(|r| r["corr"].as_i64().unwrap_or(0))
            .sum();
        let total_wrong: i64 = category_record
            .values()
            .map(|r| r["wrong"].as_i64().unwrap_or(0))
            .sum();
        let overall_accuracy = if total_correct + total_wrong > 0 {
            total_correct as f64 / (total_correct + total_wrong) as f64
        } else {
            0.0
        };

        Ok(serde_json::json!({
            "accuracy": overall_accuracy,
            "results_by_subject": category_record,
            "total_questions": total_questions,
            "output_tokens": total_output_tokens,
            "thinking_tokens": total_thinking_tokens,
        }))
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
