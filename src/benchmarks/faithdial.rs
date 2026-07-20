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

pub struct FaithDialBenchmark;

static DATASET: OnceLock<Vec<FaithDialItem>> = OnceLock::new();

#[derive(Debug, Clone, Deserialize)]
struct FaithDialItem {
    context: String,  // dialogue with background knowledge
    response: String, // assistant response
    label: String,    // "1" (hallucinated) or "0" (faithful)
}

fn load_faithdial_dataset() -> &'static Vec<FaithDialItem> {
    DATASET.get_or_init(|| {
        let cache_dir = dirs::cache_dir()
            .unwrap_or_default()
            .join("llm-benchmark-runner")
            .join("faithdial");
        let path = cache_dir.join("faithdial.json");

        if path.exists() {
            let content = fs::read_to_string(&path).expect("Failed to read cached FaithDial");
            return serde_json::from_str(&content).expect("Failed to parse FaithDial");
        }

        fs::create_dir_all(&cache_dir).expect("Failed to create cache dir");
        println!("  Downloading FaithDial dataset...");
        let url =
            "https://huggingface.co/datasets/dataframer/faithdial/resolve/main/faithdial.json";
        let bytes = download_with_retry_bytes(url, 3, 60, "llm-benchmark-runner")
            .expect("Failed to download FaithDial");

        let items: Vec<FaithDialItem> =
            serde_json::from_slice(&bytes).expect("Failed to parse FaithDial");
        fs::write(&path, bytes).expect("Failed to save FaithDial");
        items
    })
}

impl Benchmark for FaithDialBenchmark {
    fn name(&self) -> &str {
        "faithdial"
    }

    fn display_name(&self) -> &'static str {
        "FaithDial"
    }

    fn category(&self) -> BenchmarkCategory {
        BenchmarkCategory::Hallucination
    }

    fn pre_execute(&self, _config: &yaml_serde::Value) -> Result<()> {
        let _ = load_faithdial_dataset();
        Ok(())
    }

    fn execute(&self, model: &Model, _config: &yaml_serde::Value) -> Result<BenchmarkResult> {
        let dataset = load_faithdial_dataset();
        let client = Client::new_with_model_params(&model.proxy, model.set_params.as_ref())?;

        let system_prompt = "You are a factuality evaluator for dialogues. Given a dialogue with background knowledge and an assistant response, determine if the response hallucinates information not supported by the dialogue. Respond with 'hallucinated' or 'not_hallucinated'.";

        // 8-shot prompt
        let user_prompt = r#"Dialogue: What's the weather like today?
Response: It's sunny.
Label: not_hallucinated

Dialogue: I booked a flight to Paris.
Response: Your flight is to London.
Label: hallucinated

Dialogue: I'm going to the gym at 5pm.
Response: You're going to the gym.
Label: not_hallucinated

Dialogue: My mother's name is Jane.
Response: Your mother's name is Mary.
Label: hallucinated

Dialogue: I like chocolate.
Response: You like vanilla.
Label: hallucinated

Dialogue: The meeting is at 3pm.
Response: The meeting is at 3pm.
Label: not_hallucinated

Dialogue: I work as a teacher.
Response: You work as an engineer.
Label: hallucinated

Dialogue: The capital of France is Paris.
Response: The capital of France is Paris.
Label: not_hallucinated

Now, evaluate the following:

Dialogue: {context}
Response: {response}
Label:"#;

        let total = dataset.len();
        let mut correct = 0;
        let mut output_tokens_total: i64 = 0;
        let mut thinking_tokens_total: i64 = 0;

        for item in dataset {
            let prompt = user_prompt
                .replace("{context}", &item.context)
                .replace("{response}", &item.response);

            let (response, output_tokens, thinking_tokens) =
                client.chat_completion(&model.model_name, system_prompt, &prompt)?;

            output_tokens_total += output_tokens.unwrap_or(0) as i64;
            thinking_tokens_total += thinking_tokens.unwrap_or(0) as i64;

            let response = response.trim().to_lowercase();
            let is_correct = match item.label.as_str() {
                "1" => response.contains("hallucinated"),
                "0" => response.contains("not_hallucinated"),
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
                    "FaithDial: {}/{} correct ({:.1}%)",
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
                    "FaithDial: {}/{} correct ({:.1}%)",
                    correct,
                    total,
                    accuracy * 100.0
                ),
            }],
            raw: raw.clone(),
        })
    }
}
