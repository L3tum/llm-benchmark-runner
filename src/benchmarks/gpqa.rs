use crate::benchmarks::answer_classifier::{classify_wrong_answer, WrongAnswerClass};
use crate::benchmarks::Benchmark;
use crate::config::Model;
use crate::shared::{BenchmarkCategory, BenchmarkResult, TaskResult};
use crate::token_tracker::TokenTracker;
use anyhow::Result;
use regex::Regex;
use std::collections::{BTreeMap, HashMap};
use std::fs;
use std::path::PathBuf;
use std::sync::Mutex;

/// Single GPQA item from the CSV dataset.
#[derive(Debug, Clone)]
pub struct GpqaItem {
    pub question: String,
    pub options: Vec<String>,
    pub answer: String,
    pub category: String,
}

pub struct GpqaBenchmark {
    state: Mutex<GpqaState>,
}

struct GpqaState {
    items: Vec<GpqaItem>,
    current_idx: usize,
    wrong_classes: BTreeMap<WrongAnswerClass, i64>,
}

impl Default for GpqaBenchmark {
    fn default() -> Self {
        Self {
            state: Mutex::new(GpqaState {
                items: Vec::new(),
                current_idx: 0,
                wrong_classes: BTreeMap::new(),
            }),
        }
    }
}

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

impl Benchmark for GpqaBenchmark {
    fn name(&self) -> &str {
        "gpqa"
    }

    fn display_name(&self) -> &'static str {
        "GPQA"
    }

    fn category(&self) -> BenchmarkCategory {
        BenchmarkCategory::Knowledge
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

        let config_split = config
            .get("split")
            .and_then(|v| v.as_str())
            .unwrap_or("diamond");
        let data_path = self.download_dataset(config_split)?;
        let all_items = load_csv_data(&data_path)?;
        let all_data = group_by_category(all_items);

        // Determine which categories to evaluate
        let available_categories: Vec<String> = all_data.keys().cloned().collect();
        let subjects_to_eval: Vec<String> = if let Some(subj) = &subjects {
            let mut result = Vec::new();
            for s in subj {
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

        // Flatten categories into a single list
        let mut items = Vec::new();
        for category in &subjects_to_eval {
            if let Some(questions) = all_data.get(category) {
                let questions = match num_samples {
                    Some(n) if questions.len() > n as usize => questions[..n as usize].to_vec(),
                    _ => questions.clone(),
                };
                items.extend(questions);
            }
        }

        println!(
            "Evaluating GPQA: {} total questions (zero-shot CoT)",
            items.len()
        );

        let mut state = self.state.lock().expect(crate::shared::MUTEX_PANIC_MSG);
        state.items = items;
        state.current_idx = 0;
        state.wrong_classes = BTreeMap::new();
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

        let category = &q.category;
        let choice_map = "ABCD";
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

        let response = tracker.chat_completion(&model.model_name, "", &prompt)?;
        let pred = extract_answer(&response);
        let expected = q.answer.chars().next();
        let is_correct = pred == expected;

        if !is_correct {
            let wrong_class =
                classify_wrong_answer(&response, &question_text, expected.unwrap_or('?'), pred);
            let mut state = self.state.lock().expect(crate::shared::MUTEX_PANIC_MSG);
            let counter = state.wrong_classes.entry(wrong_class).or_insert(0);
            *counter += 1;
        }

        Ok(Some(
            TaskResult::new(
                format!("task-{}", idx),
                is_correct,
                if is_correct { 1.0 } else { 0.0 },
                vec![category.clone()],
            )
            .with_metadata(Some(serde_json::json!({
                "question": question_text,
                "expected": expected.map(|c| c.to_string()),
                "predicted": pred.map(|c| c.to_string()),
                "correct": is_correct,
            }))),
        ))
    }

    fn to_report_result(&self, b: &BenchmarkResult) -> Result<BenchmarkResult> {
        let raw = &b.raw;
        use crate::shared::{BreakdownTable, Score, ScoreUnit};

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

        // Build subject breakdown from per_task categories
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

        // Parse error classification from raw JSON (display names → typed enum)
        let error_classification: BTreeMap<WrongAnswerClass, i64> = raw
            .get("error_classification")
            .and_then(|v| v.as_object())
            .map(|obj| {
                obj.iter()
                    .filter_map(|(key, val)| {
                        let class = match key.as_str() {
                            "Wrong Answer Key" => Some(WrongAnswerClass::WrongAnswerKey),
                            "Invalid Answer Key" => Some(WrongAnswerClass::InvalidAnswerKey),
                            "No Answer" => Some(WrongAnswerClass::NoAnswer),
                            "Uncertainty" => Some(WrongAnswerClass::Uncertainty),
                            "Refused" => Some(WrongAnswerClass::Refused),
                            "Looping" => Some(WrongAnswerClass::Looping),
                            "Truncated" => Some(WrongAnswerClass::Truncated),
                            "Off-Topic / Hallucination" => Some(WrongAnswerClass::OffTopic),
                            _ => None,
                        };
                        class.zip(val.as_i64())
                    })
                    .collect()
            })
            .unwrap_or_default();

        Ok(BenchmarkResult {
            scores,
            breakdowns,
            error_classification,
            artifacts: vec![],
            diagnostics: vec![],
            raw: raw.clone(),
        })
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
