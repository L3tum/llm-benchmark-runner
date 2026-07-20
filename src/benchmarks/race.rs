use crate::benchmarks::Benchmark;
use crate::client::Client;
use crate::config::Model;
use crate::download::download_with_retry_bytes;
use crate::reports::model::{BenchmarkCategory, BenchmarkResult, Score, ScoreUnit};
use anyhow::Result;
use serde::Deserialize;
use std::collections::BTreeMap;
use std::fs;
use std::sync::OnceLock;

pub struct RaceBenchmark;

static DATASET: OnceLock<Vec<RaceItem>> = OnceLock::new();

#[derive(Debug, Clone, Deserialize)]
struct RaceItem {
    passage: String,
    question: String,
    option_a: String,
    option_b: String,
    option_c: String,
    option_d: String,
    answer: String,
}

fn load_race_dataset() -> &'static Vec<RaceItem> {
    DATASET.get_or_init(|| {
        let cache_dir = dirs::cache_dir()
            .unwrap_or_default()
            .join("llm-benchmark-runner")
            .join("race");
        let path = cache_dir.join("test.csv");

        if path.exists() {
            let content = fs::read_to_string(&path).expect("Failed to read cached RACE");
            let mut reader = csv::ReaderBuilder::new()
                .delimiter(b',')
                .has_headers(true)
                .from_reader(content.as_bytes());
            let mut items = Vec::new();
            for record in reader.records().flatten() {
                if record.len() >= 7 {
                    items.push(RaceItem {
                        passage: record.get(0).unwrap_or("").to_string(),
                        question: record.get(1).unwrap_or("").to_string(),
                        option_a: record.get(2).unwrap_or("").to_string(),
                        option_b: record.get(3).unwrap_or("").to_string(),
                        option_c: record.get(4).unwrap_or("").to_string(),
                        option_d: record.get(5).unwrap_or("").to_string(),
                        answer: record.get(6).unwrap_or("").to_string(),
                    });
                }
            }
            return items;
        }

        fs::create_dir_all(&cache_dir).expect("Failed to create cache dir");
        println!("  Downloading RACE dataset...");
        let url = "https://huggingface.co/datasets/ehovy/race/resolve/main/test.csv";
        let bytes = download_with_retry_bytes(url, 3, 60, "llm-benchmark-runner")
            .expect("Failed to download RACE");

        let content = String::from_utf8(Vec::from(bytes.as_ref())).expect("Failed to decode UTF-8");
        let mut reader = csv::ReaderBuilder::new()
            .delimiter(b',')
            .has_headers(true)
            .from_reader(content.as_bytes());
        let mut items = Vec::new();
        for record in reader.records().flatten() {
            if record.len() >= 7 {
                items.push(RaceItem {
                    passage: record.get(0).unwrap_or("").to_string(),
                    question: record.get(1).unwrap_or("").to_string(),
                    option_a: record.get(2).unwrap_or("").to_string(),
                    option_b: record.get(3).unwrap_or("").to_string(),
                    option_c: record.get(4).unwrap_or("").to_string(),
                    option_d: record.get(5).unwrap_or("").to_string(),
                    answer: record.get(6).unwrap_or("").to_string(),
                });
            }
        }

        fs::write(&path, &bytes).expect("Failed to save RACE");
        items
    })
}

impl Benchmark for RaceBenchmark {
    fn name(&self) -> &str {
        "race"
    }

    fn display_name(&self) -> &'static str {
        "RACE (Reading Comprehension)"
    }

    fn category(&self) -> BenchmarkCategory {
        BenchmarkCategory::Hallucination
    }

    fn pre_execute(&self, _config: &yaml_serde::Value) -> Result<()> {
        let _ = load_race_dataset();
        Ok(())
    }

    fn execute(&self, model: &Model, _config: &yaml_serde::Value) -> Result<BenchmarkResult> {
        let dataset = load_race_dataset();
        let client = Client::new_with_model_params(&model.proxy, model.set_params.as_ref())?;

        let system_prompt = "You are a reading comprehension expert. Read the passage carefully and answer the multiple-choice question based ONLY on the passage. Do not use external knowledge.";

        let user_prompt = r#"Passage: The cat sat on the mat.
Question: What did the cat sit on?
A. The bed
B. The mat
C. The table
D. The chair
Answer: B

Passage: {passage}
Question: {question}
A. {option_a}
B. {option_b}
C. {option_c}
D. {option_d}
Answer:"#;

        let total = dataset.len();
        let mut correct = 0;
        let mut output_tokens_total: i64 = 0;
        let mut thinking_tokens_total: i64 = 0;

        for item in dataset {
            let prompt = user_prompt
                .replace("{passage}", &item.passage)
                .replace("{question}", &item.question)
                .replace("{option_a}", &item.option_a)
                .replace("{option_b}", &item.option_b)
                .replace("{option_c}", &item.option_c)
                .replace("{option_d}", &item.option_d);

            let (response, output_tokens, thinking_tokens) =
                client.chat_completion(&model.model_name, system_prompt, &prompt)?;

            output_tokens_total += output_tokens.unwrap_or(0) as i64;
            thinking_tokens_total += thinking_tokens.unwrap_or(0) as i64;

            let response = response.trim().to_ascii_uppercase();
            let is_correct = response.starts_with(&item.answer.to_ascii_uppercase());

            if is_correct {
                correct += 1;
            }
        }

        let accuracy = correct as f64 / total as f64;
        let raw_json = serde_json::json!({
            "accuracy": accuracy,
            "total": total,
            "correct": correct,
            "output_tokens": output_tokens_total,
            "thinking_tokens": thinking_tokens_total,
        });

        Ok(BenchmarkResult {
            scores: BTreeMap::new(),
            breakdowns: BTreeMap::new(),
            error_classification: BTreeMap::new(),
            artifacts: vec![],
            diagnostics: vec![crate::reports::model::Diagnostic {
                level: "info".to_string(),
                message: format!(
                    "RACE (Reading Comprehension — Hallucination category): {}/{} correct ({:.1}%)",
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
            breakdowns: BTreeMap::new(),
            error_classification: BTreeMap::new(),
            artifacts: vec![],
            diagnostics: vec![crate::reports::model::Diagnostic {
                level: "info".to_string(),
                message: format!(
                    "RACE (Reading Comprehension — Hallucination category): {}/{} correct ({:.1}%)",
                    correct,
                    total,
                    accuracy * 100.0
                ),
            }],
            raw: raw.clone(),
        })
    }
}
