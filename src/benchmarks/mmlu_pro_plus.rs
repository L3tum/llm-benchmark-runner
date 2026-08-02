use crate::benchmarks::Benchmark;
use crate::token_tracker::TokenTracker;

use crate::config::Model;
use crate::download::download_with_retry_bytes;
use crate::shared::{BenchmarkCategory, BenchmarkResult, Score, ScoreUnit, TaskResult};
use anyhow::Result;
use serde::Deserialize;
use std::collections::BTreeMap;
use std::fs;
use std::sync::Mutex;

pub struct MmluProPlusBenchmark {
    state: Mutex<MmluProPlusState>,
}

struct MmluProPlusState {
    items: Vec<MmluProPlusItem>,
    current_idx: usize,
}

impl Default for MmluProPlusBenchmark {
    fn default() -> Self {
        Self {
            state: Mutex::new(MmluProPlusState {
                items: Vec::new(),
                current_idx: 0,
            }),
        }
    }
}

#[derive(Debug, Clone, Deserialize)]
#[allow(dead_code)] // id and category kept for schema alignment
struct MmluProPlusItem {
    id: String,
    category: String,
    question: String,
    choices: Vec<String>,
    choices_correct_mask: Vec<bool>,
    subject: String,
}

fn load_mmlu_pro_plus() -> Vec<MmluProPlusItem> {
    let cache_dir = dirs::cache_dir()
        .unwrap_or_default()
        .join("llm-benchmark-runner")
        .join("mmlu_pro_plus");
    let path = cache_dir.join("test.csv");

    if path.exists() {
        let content = fs::read_to_string(&path).expect("Failed to read cached MMLU-Pro+");
        return serde_json::from_str(&content).expect("Failed to parse MMLU-Pro+");
    }

    fs::create_dir_all(&cache_dir).expect("Failed to create cache dir");
    println!("  Downloading MMLU-Pro+ dataset...");
    let url = "https://huggingface.co/datasets/li-lab/MMLU-Pro+/resolve/main/test.csv";
    let bytes = download_with_retry_bytes(url, 3, 60, "llm-benchmark-runner")
        .expect("Failed to download MMLU-Pro+");

    // Parse CSV using csv crate for proper quoted field handling
    let content = String::from_utf8(Vec::from(bytes.as_ref())).expect("Failed to decode UTF-8");
    let mut reader = csv::ReaderBuilder::new()
        .delimiter(b',')
        .has_headers(true)
        .from_reader(content.as_bytes());
    let mut items = Vec::new();
    for record in reader.records().flatten() {
        let id = record.get(0).unwrap_or("").to_string();
        let category = record.get(1).unwrap_or("").to_string();
        let question = record.get(2).unwrap_or("").to_string();
        // Choices are 10 strings
        let choices: Vec<String> = (3..13)
            .filter_map(|i| record.get(i))
            .map(|s| s.to_string())
            .collect();
        // Correct mask is 10 booleans
        let correct_mask: Vec<bool> = (13..23)
            .filter_map(|i| record.get(i))
            .map(|s| s == "true")
            .collect();
        let subject = record.get(23).unwrap_or("").to_string();
        items.push(MmluProPlusItem {
            id,
            category,
            question,
            choices,
            choices_correct_mask: correct_mask,
            subject,
        });
    }

    fs::write(&path, &bytes).expect("Failed to save MMLU-Pro+");
    items
}

impl Benchmark for MmluProPlusBenchmark {
    fn name(&self) -> &str {
        "mmlu_pro_plus"
    }

    fn display_name(&self) -> &'static str {
        "MMLU-Pro+"
    }

    fn category(&self) -> BenchmarkCategory {
        BenchmarkCategory::Knowledge
    }

    fn pre_execute(&self, _config: &yaml_serde::Value) -> Result<()> {
        let items = load_mmlu_pro_plus();
        println!("MMLU-Pro+: {} total questions", items.len());
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
        use std::collections::HashSet;

        let (item, idx) = {
            let mut state = self.state.lock().expect(crate::shared::MUTEX_PANIC_MSG);
            if state.current_idx >= state.items.len() {
                return Ok(None);
            }
            let idx = state.current_idx;
            let item = state.items[idx].clone();
            state.current_idx += 1;
            (item, idx)
        };

        let system_prompt = "You are a multiple-choice question expert. For each question, select ALL correct answers from the 10 options. List the letter(s) of each correct answer (e.g., 'ABD').";

        let user_prompt = r#"Question: What is 2+2?
A. 3
B. 4
C. 5
D. 6
E. 7
F. 8
G. 9
H. 10
I. 11
J. 12
Answer: B

Question: Which of these are prime numbers?
A. 2
B. 3
C. 4
D. 5
E. 6
F. 7
G. 8
H. 9
I. 10
J. 11
Answer: ABDF

Question: {question}
{choices}
Answer:"#;

        let mut choices_str = String::new();
        let labels: Vec<char> = ('A'..='J').collect();
        for (i, choice) in item.choices.iter().enumerate() {
            choices_str.push_str(&format!(
                "{}. {}
",
                labels[i], choice
            ));
        }

        let prompt = user_prompt
            .replace("{question}", &item.question)
            .replace("{choices}", &choices_str);

        let response = tracker.chat_completion(&model.model_name, system_prompt, &prompt)?;

        let model_selected: HashSet<char> = response
            .chars()
            .filter(|c| c.is_ascii_alphabetic() && *c >= 'A' && *c <= 'J')
            .map(|c| c.to_ascii_uppercase())
            .collect();

        let expected_set: HashSet<char> = item
            .choices_correct_mask
            .iter()
            .enumerate()
            .filter_map(|(i, &correct)| if correct { Some(labels[i]) } else { None })
            .collect();

        let is_correct = model_selected == expected_set;

        Ok(Some(
            TaskResult::new(
                format!("task-{}", idx),
                is_correct,
                if is_correct { 1.0 } else { 0.0 },
                vec![item.subject.clone()],
            )
            .with_metadata(Some(serde_json::json!({
                "question": item.question,
                "correct": is_correct,
            }))),
        ))
    }

    fn to_report_result(&self, b: &BenchmarkResult) -> Result<BenchmarkResult> {
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
                    raw.get("total").and_then(|v| v.as_i64()).unwrap_or(0),
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
            for (subject, (corr, t)) in cat_map {
                let acc = if t > 0 { corr as f64 / t as f64 } else { 0.0 };
                subject_rows.insert(
                    subject,
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
        if !subject_rows.is_empty() {
            breakdowns.insert(
                "subjects".to_string(),
                crate::reports::model::BreakdownTable {
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
