use crate::benchmarks::Benchmark;
use crate::client::Client;
use crate::config::Model;
use crate::download::download_with_retry_bytes;
use crate::reports::model::{BenchmarkCategory, BenchmarkResult, Score, ScoreUnit};
use anyhow::Result;
use serde::Deserialize;
use std::collections::{BTreeMap, HashSet};
use std::fs;
use std::sync::OnceLock;

pub struct MmluProPlusBenchmark;

static DATASET: OnceLock<Vec<MmluProPlusItem>> = OnceLock::new();

#[derive(Debug, Clone, Deserialize)]
struct MmluProPlusItem {
    id: String,
    category: String,
    question: String,
    choices: Vec<String>,
    choices_correct_mask: Vec<bool>,
    subject: String,
}

fn load_mmlu_pro_plus() -> &'static Vec<MmluProPlusItem> {
    DATASET.get_or_init(|| {
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
    })
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
        let _ = load_mmlu_pro_plus();
        Ok(())
    }

    fn execute(&self, model: &Model, _config: &yaml_serde::Value) -> Result<BenchmarkResult> {
        let dataset = load_mmlu_pro_plus();
        let client = Client::new_with_model_params(&model.proxy, model.set_params.as_ref())?;

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

        let total = dataset.len();
        let mut correct = 0;
        let mut output_tokens_total: i64 = 0;
        let mut thinking_tokens_total: i64 = 0;
        let mut subject_stats: BTreeMap<String, Vec<bool>> = BTreeMap::new();

        for item in dataset {
            let mut choices_str = String::new();
            let labels: Vec<char> = ('A'..='J').collect();
            for (i, choice) in item.choices.iter().enumerate() {
                choices_str.push_str(&format!("{}. {}\n", labels[i], choice));
            }

            let prompt = user_prompt
                .replace("{question}", &item.question)
                .replace("{choices}", &choices_str);

            let (response, output_tokens, thinking_tokens) =
                client.chat_completion(&model.model_name, system_prompt, &prompt)?;

            output_tokens_total += output_tokens.unwrap_or(0) as i64;
            thinking_tokens_total += thinking_tokens.unwrap_or(0) as i64;

            // Parse model's response: extract letters A-J as a set
            let model_selected: HashSet<char> = response
                .chars()
                .filter(|c| c.is_ascii_alphabetic() && *c >= 'A' && *c <= 'J')
                .map(|c| c.to_ascii_uppercase())
                .collect();

            // Build expected answer set from correct mask
            let expected_set: HashSet<char> = item
                .choices_correct_mask
                .iter()
                .enumerate()
                .filter_map(|(i, &correct)| if correct { Some(labels[i]) } else { None })
                .collect();

            // Use set equality for correct scoring (handles extra text, commas, ordering, etc.)
            let is_correct = model_selected == expected_set;

            if is_correct {
                correct += 1;
            }

            subject_stats
                .entry(item.subject.clone())
                .or_default()
                .push(is_correct);
        }

        let accuracy = correct as f64 / total as f64;

        // Subject breakdown
        let mut breakdowns = BTreeMap::new();
        for (subject, results) in subject_stats {
            let subject_correct: i64 = results.iter().filter(|&&c| c).count() as i64;
            let subject_total = results.len() as i64;
            let subject_str = subject.clone();
            breakdowns.insert(
                subject_str.clone(),
                crate::reports::model::BreakdownTable {
                    title: subject_str,
                    rows: BTreeMap::from_iter([
                        (
                            "accuracy".to_string(),
                            BTreeMap::from_iter([(
                                "accuracy".to_string(),
                                Score::float(
                                    subject_correct as f64 / subject_total as f64 * 100.0,
                                    ScoreUnit::Percent,
                                ),
                            )]),
                        ),
                        (
                            "count".to_string(),
                            BTreeMap::from_iter([(
                                "total".to_string(),
                                Score::integer(subject_total, ScoreUnit::Count),
                            )]),
                        ),
                    ]),
                },
            );
        }

        let raw_json = serde_json::json!({
            "accuracy": accuracy,
            "total": total,
            "correct": correct,
            "output_tokens": output_tokens_total,
            "thinking_tokens": thinking_tokens_total,
        });

        Ok(BenchmarkResult {
            scores: BTreeMap::new(),
            breakdowns,
            error_classification: BTreeMap::new(),
            artifacts: vec![],
            diagnostics: vec![crate::reports::model::Diagnostic {
                level: "info".to_string(),
                message: format!(
                    "MMLU-Pro+: {}/{} correct ({:.1}%)",
                    correct,
                    total,
                    accuracy * 100.0
                ),
            }],
            raw: raw_json,
        })
    }

    fn to_report_result(&self, b: &BenchmarkResult) -> Result<BenchmarkResult> {
        let raw = &b.raw;
        let accuracy = raw.get("accuracy").and_then(|v| v.as_f64()).unwrap_or(0.0);
        let total = raw.get("total").and_then(|v| v.as_i64()).unwrap_or(0);
        let correct = raw.get("correct").and_then(|v| v.as_i64()).unwrap_or(0);
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
            breakdowns: b.breakdowns.clone(),
            error_classification: BTreeMap::new(),
            artifacts: vec![],
            diagnostics: vec![crate::reports::model::Diagnostic {
                level: "info".to_string(),
                message: format!(
                    "MMLU-Pro+: {}/{} correct ({:.1}%)",
                    correct,
                    total,
                    accuracy * 100.0
                ),
            }],
            raw: raw.clone(),
        })
    }
}
