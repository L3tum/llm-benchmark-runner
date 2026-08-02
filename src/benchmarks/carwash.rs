use crate::benchmarks::Benchmark;
use crate::config::Model;
use crate::shared::{BenchmarkResult, Score, ScoreUnit, TaskResult};
use crate::token_tracker::TokenTracker;
use anyhow::Result;
use std::collections::BTreeMap;
use std::sync::Mutex;

pub struct CarwashBenchmark {
    state: Mutex<CarwashState>,
}

struct CarwashState {
    done: bool,
}

impl Default for CarwashBenchmark {
    fn default() -> Self {
        Self {
            state: Mutex::new(CarwashState { done: false }),
        }
    }
}

const CARWASH_PROMPT: &str = r#"You are in a parking lot. The nearest car wash is 50 meters away from your current location. You have your car keys in your hand. Your car is parked right next to you. Should you walk or drive to the car wash? Explain your reasoning briefly, then give a clear answer."#;

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

    fn execute_one(
        &self,
        model: &Model,
        _config: &yaml_serde::Value,
        tracker: &mut TokenTracker,
    ) -> Result<Option<TaskResult>> {
        let should_execute = {
            let mut state = self.state.lock().expect(crate::shared::MUTEX_PANIC_MSG);
            if state.done {
                return Ok(None);
            }
            state.done = true;
            true
        };

        if !should_execute {
            return Ok(None);
        }

        let response = tracker.chat_completion(&model.model_name, "", CARWASH_PROMPT)?;

        // Pass/fail: must contain "drive" and NOT contain "walk"
        let response_lower = response.to_lowercase();
        let contains_drive = response_lower.contains("drive");
        let rejects_walking = response_lower.contains("not walk")
            || response_lower.contains("don't walk")
            || response_lower.contains("shouldn't walk")
            || response_lower.contains("don't want to walk");
        let pass = contains_drive || rejects_walking;

        Ok(Some(
            TaskResult::new("task-0", pass, if pass { 1.0 } else { 0.0 }, vec![]).with_metadata(
                Some(serde_json::json!({
                    "pass": pass,
                    "response": response,
                    "contains_drive": contains_drive,
                    "rejects_walking": rejects_walking,
                })),
            ),
        ))
    }

    fn to_report_result(&self, b: &BenchmarkResult) -> Result<BenchmarkResult> {
        let raw = &b.raw;

        // Read from per_task array (single entry)
        let (pass, output_tokens, thinking_tokens, response) = {
            if let Some(per_task) = raw.get("per_task").and_then(|v| v.as_array()) {
                if let Some(task) = per_task.first() {
                    let p = task
                        .get("passed")
                        .and_then(|v| v.as_bool())
                        .unwrap_or(false);
                    let out = task
                        .get("output_tokens")
                        .and_then(|v| v.as_i64())
                        .unwrap_or(0);
                    let think = task
                        .get("thinking_tokens")
                        .and_then(|v| v.as_i64())
                        .unwrap_or(0);
                    let resp = task
                        .get("metadata")
                        .and_then(|v| v.get("response"))
                        .and_then(|v| v.as_str())
                        .unwrap_or("");
                    (p, out, think, resp.to_string())
                } else {
                    (false, 0, 0, String::new())
                }
            } else {
                // Fallback for deserialized results
                (
                    raw.get("pass").and_then(|v| v.as_bool()).unwrap_or(false),
                    raw.get("output_tokens")
                        .and_then(|v| v.as_i64())
                        .unwrap_or(0),
                    raw.get("thinking_tokens")
                        .and_then(|v| v.as_i64())
                        .unwrap_or(0),
                    raw.get("response")
                        .and_then(|v| v.as_str())
                        .unwrap_or("")
                        .to_string(),
                )
            }
        };

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
            error_classification: BTreeMap::new(),
            artifacts: vec![],
            diagnostics: vec![crate::reports::model::Diagnostic {
                level: "info".to_string(),
                message: format!("Carwash response: {}", response),
            }],
            raw: raw.clone(),
        })
    }
}
