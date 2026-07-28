use crate::benchmarks::Benchmark;
use crate::config::Model;
use crate::download::download_with_retry_bytes;
use crate::reports::model::{BenchmarkCategory, BenchmarkResult, TaskResult};
use crate::token_tracker::TokenTracker;
use anyhow::Result;
use regex::Regex;
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, HashMap};
use std::fs;
use std::path::PathBuf;
use std::sync::Mutex;

#[derive(Debug, Deserialize, Serialize, Clone)]
pub struct MmluItem {
    pub question: String,
    pub options: Vec<String>,
    pub cot_content: Option<String>,
    pub answer: String,
    pub category: String,
}

pub struct MmluProBenchmark {
    state: Mutex<MmluProState>,
}

struct MmluProState {
    items: Vec<MmluItem>,
    current_idx: usize,
}

impl Default for MmluProBenchmark {
    fn default() -> Self {
        Self {
            state: Mutex::new(MmluProState {
                items: Vec::new(),
                current_idx: 0,
            }),
        }
    }
}

impl MmluProBenchmark {
    pub fn download_dataset(&self, split: &str) -> Result<PathBuf> {
        let cache_dir = dirs::cache_dir()
            .unwrap_or_default()
            .join("llm-benchmark-runner")
            .join("mmlu_pro");
        fs::create_dir_all(&cache_dir)?;
        let path = cache_dir.join(format!("{}.json", split));
        if path.exists() {
            return Ok(path);
        }
        let url = format!(
            "https://huggingface.co/datasets/TIGER-Lab/MMLU-Pro/resolve/main/{}/{}.json",
            split, split
        );
        println!("  Downloading MMLU-Pro {} data...", split);
        let bytes = download_with_retry_bytes(&url, 3, 120, "llm-benchmark-runner")?;
        fs::write(&path, bytes)?;
        Ok(path)
    }

    pub fn load_dataset(&self, path: &PathBuf) -> Result<Vec<MmluItem>> {
        let content = fs::read_to_string(path)?;
        let items: Vec<MmluItem> = serde_json::from_str(&content)?;
        Ok(items)
    }

    fn group_by_category(items: Vec<MmluItem>) -> HashMap<String, Vec<MmluItem>> {
        let mut groups: HashMap<String, Vec<MmluItem>> = HashMap::new();
        for item in items {
            let options: Vec<String> = item.options.into_iter().filter(|o| o != "N/A").collect();
            let category = item.category.clone();
            groups
                .entry(category)
                .or_default()
                .push(MmluItem { options, ..item });
        }
        groups
    }
}

impl Benchmark for MmluProBenchmark {
    fn name(&self) -> &str {
        "mmlu_pro"
    }

    fn display_name(&self) -> &'static str {
        "MMLU-Pro"
    }

    fn category(&self) -> BenchmarkCategory {
        BenchmarkCategory::Knowledge
    }

    fn pre_execute(&self, _config: &yaml_serde::Value) -> Result<()> {
        let data_path = self.download_dataset("test")?;
        let all_items = self.load_dataset(&data_path)?;
        println!("MMLU-Pro: {} total questions", all_items.len());
        let mut state = self.state.lock().unwrap();
        state.items = all_items;
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

        let question_text = q.question.clone();
        let mut prompt = format!(
            "The following are multiple choice questions (with answers) about {}. Think step by step and then output the answer in the format of \"The answer is (X)\" at the end.\n\n",
            q.category
        );
        prompt.push_str(&format!("Question: {}\nOptions: ", question_text));
        let choice_map = "ABCDEFGH";
        for (i, opt) in q.options.iter().enumerate() {
            if i < choice_map.len() {
                prompt.push_str(&format!("{}: {}\n", &choice_map[i..i + 1], opt));
            }
        }
        prompt.push_str("Answer: ");

        let response = tracker.chat_completion(&model.model_name, "", &prompt)?;
        let pred = extract_answer(&response);
        let expected = q.answer.chars().next();
        let is_correct = pred == expected;

        Ok(Some(
            TaskResult::new(
                format!("task-{}", idx),
                is_correct,
                if is_correct { 1.0 } else { 0.0 },
                vec![q.category.clone()],
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
        use crate::reports::model::{BreakdownTable, Score, ScoreUnit};

        let raw = &b.raw;

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

        let mut category_rows = BTreeMap::new();
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
            for (cat, (corr, t)) in cat_map {
                let acc = if t > 0 { corr as f64 / t as f64 } else { 0.0 };
                category_rows.insert(
                    cat,
                    BTreeMap::from([
                        (
                            "accuracy".to_string(),
                            Score::float(acc * 100.0, ScoreUnit::Percent),
                        ),
                        (
                            "correct".to_string(),
                            Score::integer(corr, ScoreUnit::Count),
                        ),
                        (
                            "wrong".to_string(),
                            Score::integer(t - corr, ScoreUnit::Count),
                        ),
                    ]),
                );
            }
        }

        let mut breakdowns = BTreeMap::new();
        if !category_rows.is_empty() {
            breakdowns.insert(
                "categories".to_string(),
                BreakdownTable {
                    title: "Category Breakdown".to_string(),
                    rows: category_rows,
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

fn extract_answer(text: &str) -> Option<char> {
    // Scan entire text for answer patterns, use the last match
    let re1 = Regex::new(r"\banswer is\s*\(?([A-J])\)?").ok()?;
    let last = re1.captures_iter(text).last();
    if let Some(caps) = last {
        if let Some(m) = caps.get(1) {
            return m.as_str().chars().next();
        }
    }

    let re2 = Regex::new(r"[aA]nswer:\s*([A-J])").ok()?;
    let last = re2.captures_iter(text).last();
    if let Some(caps) = last {
        if let Some(m) = caps.get(1) {
            return m.as_str().chars().next();
        }
    }

    // Final: last single letter from A-J, excluding those followed by comma/semicolon and another letter (e.g., "A, B")
    let re_letter = Regex::new(r"\b([A-J])\b").ok()?;
    let re_sequence = Regex::new(r"[A-J][\s]*[,;:]\s*[A-J]").ok()?;
    let mut last_letter = None;
    for caps in re_letter.captures_iter(text) {
        if let Some(letter_match) = caps.get(1) {
            let letter_start = letter_match.start();
            // Check if this letter is part of a sequence pattern (e.g., "A, B")
            let context = &text[letter_start..text.len().min(letter_start + 6)];
            if re_sequence.find(context).is_some() {
                // This letter is part of a sequence, skip it
                continue;
            }
            last_letter = letter_match.as_str().chars().next();
        }
    }
    last_letter
}
