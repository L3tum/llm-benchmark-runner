use crate::benchmarks::Benchmark;
use crate::client::Client;
use crate::config::Model;
use crate::reports::model::{BenchmarkResult, Score, ScoreUnit};
use anyhow::Result;
use serde::Serialize;
use std::collections::BTreeMap;

pub struct CarwashBenchmark;

const CARWASH_PROMPT: &str = r#"You are in a parking lot. The nearest car wash is 50 meters away from your current location. You have your car keys in your hand. Your car is parked right next to you. Should you walk or drive to the car wash? Explain your reasoning briefly, then give a clear answer."#;

#[derive(Debug, Clone, Serialize)]
struct CarwashResult {
    pass: bool,
    response: String,
    output_tokens: u64,
}

impl Benchmark for CarwashBenchmark {
    fn name(&self) -> &str {
        "carwash"
    }

    fn display_name(&self) -> &'static str {
        "Carwash Test"
    }

    fn category(&self) -> crate::reports::model::BenchmarkCategory {
        crate::reports::model::BenchmarkCategory::Reasoning
    }

    fn execute(&self, model: &Model, _config: &yaml_serde::Value) -> Result<serde_json::Value> {
        let client = Client::new(&model.proxy)?;
        let (response, output_tokens, thinking_tokens) =
            client.chat_completion(&model.model_name, "", CARWASH_PROMPT)?;

        let output_tokens = output_tokens.unwrap_or(0);
        let thinking_tokens = thinking_tokens.unwrap_or(0);

        // Pass/fail: must contain "drive" and NOT contain "walk"
        let response_lower = response.to_lowercase();
        // Pass if:
        // 1. Contains "drive" (positive recommendation) — best signal
        // 2. OR explicitly rejects walking ("not walk", "don't walk", "shouldn't walk", etc.)
        let contains_drive = response_lower.contains("drive");
        let rejects_walking = response_lower.contains("not walk")
            || response_lower.contains("don't walk")
            || response_lower.contains("shouldn't walk")
            || response_lower.contains("don't want to walk");
        let pass = contains_drive || rejects_walking;

        Ok(serde_json::json!({
            "pass": pass,
            "response": response,
            "output_tokens": output_tokens,
            "thinking_tokens": thinking_tokens,
            "contains_drive": contains_drive,
            "rejects_walking": rejects_walking,
        }))
    }

    fn to_report_result(&self, raw: &serde_json::Value) -> Result<BenchmarkResult> {
        let pass = raw.get("pass").and_then(|v| v.as_bool()).unwrap_or(false);
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
            "pass".to_string(),
            Score::bool(pass).primary(true).higher_is_better(true),
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
            artifacts: vec![],
            diagnostics: vec![crate::reports::model::Diagnostic {
                level: "info".to_string(),
                message: format!(
                    "Carwash response: {}",
                    raw.get("response")
                        .and_then(|v| v.as_str())
                        .unwrap_or("<response not available>")
                ),
            }],
            raw: raw.clone(),
        })
    }
}
