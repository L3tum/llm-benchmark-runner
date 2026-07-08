use crate::client::Client;
use crate::config::Model;
use crate::reports::model::BenchmarkResult;
use anyhow::Result;
use regex::Regex;
use std::collections::{BTreeMap, HashMap};
use std::fs;
use std::path::PathBuf;

/// Single GPQA item from the CSV dataset.
#[derive(Debug, Clone)]
pub struct GpqaItem {
    pub question: String,
    pub options: Vec<String>,
    pub answer: String,
    pub category: String,
}

pub struct GpqaBenchmark;

fn load_csv_data(path: &PathBuf) -> Result<Vec<GpqaItem>> {
    use csv::ReaderBuilder;

    let mut items = Vec::new();

    let mut reader = ReaderBuilder::new().has_headers(true).from_path(path)?;

    for record_result in reader.records() {
        let record = record_result?;
        let question = record.get(0).unwrap_or("").trim().to_string();
        let choice1 = record.get(1).unwrap_or("").trim().to_string();
        let choice2 = record.get(2).unwrap_or("").trim().to_string();
        let choice3 = record.get(3).unwrap_or("").trim().to_string();
        let choice4 = record.get(4).unwrap_or("").trim().to_string();
        let answer = record.get(5).unwrap_or("").trim().to_string();
        let category = record.get(6).unwrap_or("").trim().to_string();

        let options = vec![choice1, choice2, choice3, choice4];
        items.push(GpqaItem {
            question,
            options,
            answer,
            category,
        });
    }
    Ok(items)
}

fn group_by_category(items: Vec<GpqaItem>) -> HashMap<String, Vec<GpqaItem>> {
    let mut groups: HashMap<String, Vec<GpqaItem>> = HashMap::new();
    for item in items {
        let category = item.category.clone();
        groups.entry(category).or_default().push(item);
    }
    groups
}

impl super::Benchmark for GpqaBenchmark {
    fn name(&self) -> &str {
        "gpqa"
    }

    fn display_name(&self) -> &'static str {
        "GPQA"
    }

    fn category(&self) -> crate::reports::model::BenchmarkCategory {
        crate::reports::model::BenchmarkCategory::Knowledge
    }

    fn pre_execute(&self, config: &serde_yaml::Value) -> Result<()> {
        // Download the diamond dataset
        let config_split = config
            .get("split")
            .and_then(|v| v.as_str())
            .unwrap_or("diamond");
        self.download_dataset(config_split)?;
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

    fn execute(&self, model: &Model, config: &serde_yaml::Value) -> Result<serde_json::Value> {
        let client = Client::new(&model.proxy)?;
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

        let config_split = config
            .get("split")
            .and_then(|v| v.as_str())
            .unwrap_or("diamond");
        let data_path = self.download_dataset(config_split)?;
        let all_items = load_csv_data(&data_path)?;
        let all_data = group_by_category(all_items);

        // Determine which categories to evaluate
        let available_categories: Vec<String> = all_data.keys().cloned().collect();
        let subjects_to_eval: Vec<String> = if let Some(subj) = subjects {
            let mut result = Vec::new();
            for s in &subj {
                if all_data.contains_key(s) {
                    result.push(s.clone());
                } else {
                    eprintln!(
                        "  WARNING: GPQA category '{}' not found, skipping. Available: {:?}",
                        s, available_categories
                    );
                }
            }
            result
        } else {
            available_categories
        };

        let mut category_record: HashMap<String, serde_json::Value> = HashMap::new();
        let mut total_questions = 0usize;
        let mut total_output_tokens: u64 = 0;
        let mut total_thinking_tokens: u64 = 0;

        let choice_map = "ABCD";

        for category in &subjects_to_eval {
            let questions = all_data
                .get(category)
                .ok_or_else(|| anyhow::anyhow!("Category {} not found", category))?
                .clone();
            let questions = match num_samples {
                Some(n) if questions.len() > n as usize => questions[..n as usize].to_vec(),
                _ => questions,
            };

            // Use zero-shot CoT (no few-shot examples available for Diamond subset)
            println!(
                "\nEvaluating {}: {} questions (zero-shot CoT)",
                category,
                questions.len()
            );

            let mut category_correct = 0usize;
            let mut category_total = 0usize;

            for q in &questions {
                let question_text = q.question.clone();
                let mut prompt = format!(
                    "The following are multiple choice questions (with answers) about {}. Think step by step and then output the answer in the format of \"The answer is (X)\" at the end.\n\n",
                    category
                );
                prompt.push_str(&format!("Question: {}\nOptions: ", question_text));
                for (i, opt) in q.options.iter().enumerate() {
                    prompt.push_str(&format!("{}: {}\n", &choice_map[i..i + 1], opt));
                }
                prompt.push_str("Answer: ");

                let (response, output_tokens, thinking_tokens) =
                    client.chat_completion(&model.model_name, "", &prompt)?;
                total_output_tokens += output_tokens.unwrap_or(0);
                total_thinking_tokens += thinking_tokens.unwrap_or(0);
                let pred = extract_answer(&response).ok_or_else(|| {
                    eprintln!("  Error extracting answer from: {}", response);
                    anyhow::anyhow!("Cannot extract answer")
                })?;
                let is_correct = pred == q.answer.chars().next().unwrap_or(pred);
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
            category_record.insert(category.clone(), serde_json::Value::Object(record));
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

impl GpqaBenchmark {
    pub fn download_dataset(&self, split: &str) -> Result<PathBuf> {
        let cache_dir = dirs::cache_dir()
            .unwrap_or_default()
            .join("llm-benchmark-runner")
            .join("gpqa");
        fs::create_dir_all(&cache_dir)?;
        let path = cache_dir.join(format!("{}.csv", split));
        if path.exists() {
            return Ok(path);
        }

        let hf_token = std::env::var("HF_TOKEN").ok();
        let url = format!(
            "https://huggingface.co/datasets/idavidrein/gpqa/resolve/main/gpqa_{}.csv",
            split
        );
        println!(
            "  Downloading GPQA {} data... (requires HF_TOKEN env variable if gated)",
            split
        );

        let request = reqwest::blocking::get(&url)?;
        // If the response is 401, try with the HF_TOKEN header
        if request.status() == 401 {
            if let Some(token) = &hf_token {
                let client = reqwest::blocking::Client::new();
                let response = client
                    .get(&url)
                    .header("Authorization", format!("Bearer {}", token))
                    .header("Accept", "text/csv")
                    .send()?;
                if response.status().is_success() {
                    let bytes = response.bytes()?;
                    fs::write(&path, bytes)?;
                    return Ok(path);
                }
            }
            // Re-try the original response's body anyway
            let bytes = request.bytes()?;
            fs::write(&path, bytes)?;
        } else {
            let bytes = request.bytes()?;
            fs::write(&path, bytes)?;
        }

        Ok(path)
    }
}

fn extract_answer(text: &str) -> Option<char> {
    // Scan entire text for answer patterns, use the last match
    let re1 = Regex::new(r"\banswer is\s*\(?([A-D])\)?").ok()?;
    let last = re1.captures_iter(text).last();
    if let Some(caps) = last {
        if let Some(m) = caps.get(1) {
            return m.as_str().chars().next();
        }
    }

    let re2 = Regex::new(r"[aA]nswer:\s*([A-D])").ok()?;
    let last = re2.captures_iter(text).last();
    if let Some(caps) = last {
        if let Some(m) = caps.get(1) {
            return m.as_str().chars().next();
        }
    }

    // Final fallback: find the last single letter from A-D that isn't part of a sequence like "A, B" or "A; B"
    let re_letter = Regex::new(r"\b([A-D])\b").ok()?;
    // Find all A-D sequence patterns (case-insensitive, with commas/semicolons)
    let re_sequence = Regex::new(r"\b[A-D]\b\s*[,;]\s*\b[A-D]\b")
        .ok()?
        .find_iter(text)
        .map(|m| (m.start(), m.end()))
        .collect::<Vec<_>>();

    let mut last_letter = None;
    for caps in re_letter.captures_iter(text) {
        if let Some(letter_match) = caps.get(1) {
            let start = letter_match.start();
            // Check if this letter position falls within any sequence range
            let in_sequence = re_sequence.iter().any(|(s, e)| start >= *s && start < *e);
            if !in_sequence {
                last_letter = letter_match.as_str().chars().next();
            }
        }
    }
    last_letter
}
