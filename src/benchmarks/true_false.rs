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

pub struct TrueFalseBenchmark;

static DATASET: OnceLock<Vec<TrueFalseItem>> = OnceLock::new();

#[derive(Debug, Clone, Deserialize)]
struct TrueFalseItem {
    statement: String,
    label: String, // "True", "False"
}

fn load_true_false_dataset() -> &'static Vec<TrueFalseItem> {
    DATASET.get_or_init(|| {
        let cache_dir = dirs::cache_dir()
            .unwrap_or_default()
            .join("llm-benchmark-runner")
            .join("true_false");
        let path = cache_dir.join("statements.json");

        if path.exists() {
            let content =
                fs::read_to_string(&path).expect("Failed to read cached True-False dataset");
            return serde_json::from_str(&content).expect("Failed to parse True-False dataset");
        }

        fs::create_dir_all(&cache_dir).expect("Failed to create cache dir");
        println!("  Downloading True-False dataset from TruthfulQA...");

        // Use the gen (generation) subset from TruthfulQA - questions that have a True/False label
        let url =
            "https://huggingface.co/datasets/truthfulqa/truthful_qa/resolve/main/generation.csv";
        let bytes = download_with_retry_bytes(url, 3, 60, "llm-benchmark-runner")
            .expect("Failed to download TruthfulQA generation dataset");

        // Parse CSV and extract statement, label pairs
        let content = String::from_utf8(Vec::from(bytes.as_ref())).expect("Failed to decode UTF-8");
        let mut items = Vec::new();
        for line in content.lines().skip(1) {
            // skip header
            if line.is_empty() {
                continue;
            }
            // CSV format: question, correct_answers, incorrect_answers, type, category, label
            let fields: Vec<&str> = line.split(",").collect();
            if fields.len() >= 6 {
                let question = fields[0].trim_matches('"').to_string();
                let label = fields[5].trim_matches('"').to_string();
                items.push(TrueFalseItem {
                    statement: question,
                    label,
                });
            }
        }

        fs::write(&path, &bytes).expect("Failed to save True-False dataset");
        items
    })
}

impl Benchmark for TrueFalseBenchmark {
    fn name(&self) -> &str {
        "true_false"
    }

    fn display_name(&self) -> &'static str {
        "True-False"
    }

    fn category(&self) -> BenchmarkCategory {
        BenchmarkCategory::Hallucination
    }

    fn pre_execute(&self, _config: &yaml_serde::Value) -> Result<()> {
        let _ = load_true_false_dataset();
        Ok(())
    }

    fn execute(&self, model: &Model, _config: &yaml_serde::Value) -> Result<BenchmarkResult> {
        let dataset = load_true_false_dataset();
        let client = Client::new_with_model_params(&model.proxy, model.set_params.as_ref())?;

        let system_prompt = "You are a factuality checker. Given a statement, determine if it is True or False based on your knowledge. Respond with only 'True' or 'False'.";

        // 8-shot prompt
        let user_prompt = r#"Statement: The capital of Australia is Sydney.
Answer: False

Statement: The Earth revolves around the Sun.
Answer: True

Statement: Humans have 46 chromosomes.
Answer: False

Statement: Albert Einstein was born in 1879.
Answer: True

Statement: The moon is made of cheese.
Answer: False

Statement: Water boils at 100 degrees Celsius at sea level.
Answer: True

Statement: The Eiffel Tower is in Paris.
Answer: True

Statement: The sun rises in the west.
Answer: False

Statement: {statement}
Answer:"#;

        let total = dataset.len();
        let mut correct = 0;
        let mut output_tokens_total: i64 = 0;
        let mut thinking_tokens_total: i64 = 0;

        for item in dataset {
            let prompt = user_prompt.replace("{statement}", &item.statement);
            let (response, output_tokens, thinking_tokens) =
                client.chat_completion(&model.model_name, system_prompt, &prompt)?;

            output_tokens_total += output_tokens.unwrap_or(0) as i64;
            thinking_tokens_total += thinking_tokens.unwrap_or(0) as i64;

            let response = response.trim().to_lowercase();
            let is_correct = match item.label.to_lowercase().as_str() {
                "true" => response.contains("true"),
                "false" => response.contains("false"),
                _ => response.contains("false"),
            };

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
                    "True-False: {}/{} correct ({:.1}%)",
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
                    "True-False: {}/{} correct ({:.1}%)",
                    correct,
                    total,
                    accuracy * 100.0
                ),
            }],
            raw: raw.clone(),
        })
    }
}
