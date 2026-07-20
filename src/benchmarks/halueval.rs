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

pub struct HaluEvalBenchmark;

static DATASET: OnceLock<Vec<HaluEvalItem>> = OnceLock::new();

#[derive(Debug, Clone, Deserialize)]
struct HaluEvalItem {
    question: String,
    context: String,
    answer: String,
    label: String, // "hallucinated" or "not_hallucinated" or similar
}

fn load_halueval_dataset() -> &'static Vec<HaluEvalItem> {
    DATASET.get_or_init(|| {
        let cache_dir = dirs::cache_dir()
            .unwrap_or_default()
            .join("llm-benchmark-runner")
            .join("halueval");
        let path = cache_dir.join("qa.json");

        if path.exists() {
            let content = fs::read_to_string(&path).expect("Failed to read cached HaluEval");
            return serde_json::from_str(&content).expect("Failed to parse HaluEval");
        }

        fs::create_dir_all(&cache_dir).expect("Failed to create cache dir");
        println!("  Downloading HaluEval dataset (requires HF_TOKEN)...");
        let url = "https://huggingface.co/datasets/marsha1908/HaluEval/resolve/main/qa.json";
        let bytes = download_with_retry_bytes(url, 3, 60, "llm-benchmark-runner")
            .expect("Failed to download HaluEval");

        let items: Vec<HaluEvalItem> =
            serde_json::from_slice(&bytes).expect("Failed to parse HaluEval");
        fs::write(&path, bytes).expect("Failed to save HaluEval");
        items
    })
}

impl Benchmark for HaluEvalBenchmark {
    fn name(&self) -> &str {
        "halueval"
    }

    fn display_name(&self) -> &'static str {
        "HaluEval (QA)"
    }

    fn category(&self) -> BenchmarkCategory {
        BenchmarkCategory::Hallucination
    }

    fn pre_execute(&self, _config: &yaml_serde::Value) -> Result<()> {
        let _ = load_halueval_dataset();
        Ok(())
    }

    fn execute(&self, model: &Model, _config: &yaml_serde::Value) -> Result<BenchmarkResult> {
        let dataset = load_halueval_dataset();
        let client = Client::new_with_model_params(&model.proxy, model.set_params.as_ref())?;

        let system_prompt = "You are a factuality evaluator. Given a question, context, and answer, determine if the answer contains hallucinated information (facts not supported by the context). Respond with 'hallucinated' or 'not_hallucinated'.";

        let user_prompt = r#"Question: What did the cat do?
Context: The cat was sleeping on the sofa.
Answer: The cat was sleeping on the sofa.
Verdict: not_hallucinated

Question: What did the cat do?
Context: The cat was sleeping on the sofa.
Answer: The cat was playing with a ball.
Verdict: hallucinated

Question: {question}
Context: {context}
Answer: {answer}
Verdict:"#;

        let total = dataset.len();
        let mut correct = 0;
        let mut output_tokens_total: i64 = 0;
        let mut thinking_tokens_total: i64 = 0;

        for item in dataset {
            let prompt = user_prompt
                .replace("{question}", &item.question)
                .replace("{context}", &item.context)
                .replace("{answer}", &item.answer);

            let (response, output_tokens, thinking_tokens) =
                client.chat_completion(&model.model_name, system_prompt, &prompt)?;

            output_tokens_total += output_tokens.unwrap_or(0) as i64;
            thinking_tokens_total += thinking_tokens.unwrap_or(0) as i64;

            let response = response.trim().to_lowercase();
            let is_correct = match item.label.as_str() {
                "hallucinated" => response.contains("hallucinated"),
                "not_hallucinated" => response.contains("not_hallucinated"),
                _ => response.contains("not_hallucinated"),
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
                    "HaluEval QA: {}/{} correct ({:.1}%)",
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
                    "HaluEval QA: {}/{} correct ({:.1}%)",
                    correct,
                    total,
                    accuracy * 100.0
                ),
            }],
            raw: raw.clone(),
        })
    }
}
