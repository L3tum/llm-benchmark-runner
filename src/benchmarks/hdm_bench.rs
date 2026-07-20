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

pub struct HdmBenchBenchmark;

static DATASET: OnceLock<Vec<HDMItem>> = OnceLock::new();

#[derive(Debug, Clone, Deserialize)]
struct HDMItem {
    prompt: String,
    context: String,
    response: String,
    label: String, // "hallucinated", "grounded", or "common_knowledge"
}

fn load_hdm_dataset() -> &'static Vec<HDMItem> {
    DATASET.get_or_init(|| {
        let cache_dir = dirs::cache_dir()
            .unwrap_or_default()
            .join("llm-benchmark-runner")
            .join("hdm_bench");
        let path = cache_dir.join("HDM-Bench.json");

        if path.exists() {
            let content = fs::read_to_string(&path).expect("Failed to read cached HDM-Bench");
            return serde_json::from_str(&content).expect("Failed to parse HDM-Bench");
        }

        fs::create_dir_all(&cache_dir).expect("Failed to create cache dir");
        println!("  Downloading HDM-Bench synthetic dataset...");
        let url =
            "https://huggingface.co/datasets/dataframer/HDM-Bench/resolve/main/HDM-Bench.json";
        let bytes = download_with_retry_bytes(url, 3, 60, "llm-benchmark-runner")
            .expect("Failed to download HDM-Bench");

        let items: Vec<HDMItem> =
            serde_json::from_slice(&bytes).expect("Failed to parse HDM-Bench");
        fs::write(&path, &bytes).expect("Failed to save HDM-Bench");
        items
    })
}

impl Benchmark for HdmBenchBenchmark {
    fn name(&self) -> &str {
        "hdm_bench"
    }

    fn display_name(&self) -> &'static str {
        "HDM-Bench (Context Hallucination)"
    }

    fn category(&self) -> BenchmarkCategory {
        BenchmarkCategory::Hallucination
    }

    fn pre_execute(&self, _config: &yaml_serde::Value) -> Result<()> {
        let _ = load_hdm_dataset();
        Ok(())
    }

    fn execute(&self, model: &Model, _config: &yaml_serde::Value) -> Result<BenchmarkResult> {
        let dataset = load_hdm_dataset();
        let client = Client::new_with_model_params(&model.proxy, model.set_params.as_ref())?;

        let system_prompt = "You are a factuality evaluator. Given a prompt, context, and response, determine if the response is hallucinated (contains information not grounded in the context), grounded (fully supported by the context), or common knowledge (factual information not in the context but universally known). Respond with 'hallucinated', 'grounded', or 'common_knowledge'.";

        // Few-shot examples (simplified)
        let user_prompt = r#"Prompt: What is the capital of France?
Context: [empty]
Response: Paris.
Label: grounded

Prompt: Who wrote this paper?
Context: This paper was written by Alice and Bob.
Response: Charlie wrote the paper.
Label: hallucinated

Prompt: What is the speed of light?
Context: The study discusses quantum entanglement.
Response: The speed of light is 299,792,458 m/s.
Label: common_knowledge

Prompt: What did Alice say?
Context: Alice said the meeting is at 3pm.
Response: Alice said the meeting is at 5pm.
Label: hallucinated

Prompt: {prompt}
Context: {context}
Response: {response}
Label:"#;

        let total = dataset.len();
        let mut correct = 0;
        let mut output_tokens_total: i64 = 0;
        let mut thinking_tokens_total: i64 = 0;

        for item in dataset {
            let prompt = user_prompt
                .replace("{prompt}", &item.prompt)
                .replace("{context}", &item.context)
                .replace("{response}", &item.response);

            let (response, output_tokens, thinking_tokens) =
                client.chat_completion(&model.model_name, system_prompt, &prompt)?;

            output_tokens_total += output_tokens.unwrap_or(0) as i64;
            thinking_tokens_total += thinking_tokens.unwrap_or(0) as i64;

            let response = response.trim().to_lowercase();
            let expected = item.label.to_lowercase();
            // Use word-boundary matching to avoid false positives (e.g., "ground" matching "grounded")
            // For "common_knowledge", also accept "common-knowledge" and "common knowledge"
            let is_correct = if expected == "hallucinated" {
                response.contains("hallucinat") // matches "hallucinated", "hallucination"
            } else if expected == "grounded" {
                response.contains("grounded")
            } else if expected == "common_knowledge" {
                response.contains("common.knowledge")
                    || response.contains("common-knowledge")
                    || response.contains("common knowledge")
            } else {
                // For any unexpected label, report as incorrect (don't default to common_knowledge)
                false
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
                    "HDM-Bench: {}/{} correct ({:.1}%)",
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
                    "HDM-Bench: {}/{} correct ({:.1}%)",
                    correct,
                    total,
                    accuracy * 100.0
                ),
            }],
            raw: raw.clone(),
        })
    }
}
