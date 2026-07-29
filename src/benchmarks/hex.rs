use crate::benchmarks::Benchmark;
use crate::config;
use crate::config::Model;
use crate::reports::model::{BenchmarkCategory, BenchmarkResult, Score, ScoreUnit, TaskResult};
use crate::token_tracker::TokenTracker;
use anyhow::Result;
use std::collections::BTreeMap;
use std::sync::Mutex;

/// Hex encoding/decoding benchmark.
/// Direct prompt variant.
pub struct HexBenchmark {
    state: Mutex<HexState>,
}

/// Hex encoding/decoding benchmark using tools.
/// Tool-calling variant.
pub struct HexToolsBenchmark {
    state: Mutex<HexState>,
}

struct HexState {
    instances: Vec<HexInstance>,
    current_idx: usize,
}

#[derive(Clone)]
struct HexInstance {
    plaintext: String,
    hex_encoded: String,
    direction: HexDirection,
}

#[derive(Clone)]
enum HexDirection {
    Encode,
    Decode,
}

const DEFAULT_PLAINTEXTS: &[&str] = &[
    "hello",
    "world",
    "test",
    "data",
    "hex",
    "encode",
    "decode",
    "pelican",
    "moonwalk",
    "benchmark",
    "science",
    "machine",
    "learning",
    "language",
    "computer",
    "developer",
    "challenge",
    "function",
    "variable",
    "constant",
    "iterator",
    "generator",
    "callback",
    "promise",
    "recursive",
];

impl Default for HexBenchmark {
    fn default() -> Self {
        Self {
            state: Mutex::new(HexState {
                instances: Vec::new(),
                current_idx: 0,
            }),
        }
    }
}

impl Default for HexToolsBenchmark {
    fn default() -> Self {
        Self {
            state: Mutex::new(HexState {
                instances: Vec::new(),
                current_idx: 0,
            }),
        }
    }
}

fn hex_encode(input: &str) -> String {
    input
        .as_bytes()
        .iter()
        .map(|b| format!("{:02x}", b))
        .collect()
}

fn hex_decode(input: &str) -> Result<String> {
    let input = input.trim().replace(" ", "");
    let mut bytes = Vec::with_capacity(input.len() / 2);
    for i in (0..input.len()).step_by(2) {
        if i + 1 >= input.len() {
            return Err(anyhow::anyhow!("Odd-length hex string"));
        }
        let byte = u8::from_str_radix(&input[i..i + 2], 16)
            .map_err(|e| anyhow::anyhow!("Invalid hex: {}", e))?;
        bytes.push(byte);
    }
    String::from_utf8(bytes).map_err(|e| anyhow::anyhow!(e))
}

fn build_instances(text_list: &[String]) -> Vec<HexInstance> {
    let mut instances = Vec::new();
    for text in text_list {
        let hex_encoded = hex_encode(text);
        instances.push(HexInstance {
            plaintext: text.clone(),
            hex_encoded: hex_encoded.clone(),
            direction: HexDirection::Encode,
        });
        instances.push(HexInstance {
            plaintext: text.clone(),
            hex_encoded,
            direction: HexDirection::Decode,
        });
    }
    instances
}

fn load_config(state: &mut HexState, config: &yaml_serde::Value) {
    let custom_texts = config::extract_string_vec(config, "text_samples");
    let num_samples = config::extract_usize(config, "num_samples");

    let text_list =
        custom_texts.unwrap_or_else(|| DEFAULT_PLAINTEXTS.iter().map(|s| s.to_string()).collect());

    state.instances = build_instances(&text_list);

    if let Some(n) = num_samples {
        if n < state.instances.len() {
            state.instances.truncate(n);
        }
    }
    state.current_idx = 0;
}

fn extract_task_stats(raw: &serde_json::Value) -> (i64, i64, i64, i64) {
    let total = raw.get("total_tasks").and_then(|v| v.as_i64()).unwrap_or(0);
    let correct = raw
        .get("passed_tasks")
        .and_then(|v| v.as_i64())
        .unwrap_or(0);
    let output_tokens = raw
        .get("output_tokens")
        .and_then(|v| v.as_i64())
        .unwrap_or(0);
    let thinking_tokens = raw
        .get("thinking_tokens")
        .and_then(|v| v.as_i64())
        .unwrap_or(0);
    (total, correct, output_tokens, thinking_tokens)
}

impl Benchmark for HexBenchmark {
    fn name(&self) -> &str {
        "hex"
    }

    fn display_name(&self) -> &'static str {
        "Hex Encoding/Decoding"
    }

    fn category(&self) -> BenchmarkCategory {
        BenchmarkCategory::StringManipulation
    }

    fn pre_execute(&self, config: &yaml_serde::Value) -> Result<()> {
        load_config(&mut self.state.lock().unwrap(), config);
        Ok(())
    }

    fn execute_one(
        &self,
        model: &Model,
        _config: &yaml_serde::Value,
        tracker: &mut TokenTracker,
    ) -> Result<Option<TaskResult>> {
        let (instance, task_id) = {
            let mut state = self.state.lock().unwrap();
            if state.current_idx >= state.instances.len() {
                return Ok(None);
            }
            let instance = state.instances[state.current_idx].clone();
            let idx = state.current_idx;
            state.current_idx += 1;
            (instance, format!("task-{}", idx))
        };

        let (prompt, expected) = match instance.direction {
            HexDirection::Encode => (
                format!("Convert this text to hex encoding: '{}'\nReturn only the hex string (lowercase, no spaces).", instance.plaintext),
                instance.hex_encoded.clone(),
            ),
            HexDirection::Decode => (
                format!("Convert this hex string to text: '{}'\nReturn only the decoded text.", instance.hex_encoded),
                instance.plaintext.clone(),
            ),
        };

        let response = tracker.chat_completion(&model.model_name, "", &prompt)?;
        let trimmed = response.trim().to_lowercase();

        // Normalize: remove spaces, colons, 0x prefix
        let normalized = trimmed
            .replace([' ', ':', '\n'], "")
            .strip_prefix("0x")
            .unwrap_or(&trimmed)
            .to_string();
        let pass = normalized == expected;

        let direction = match instance.direction {
            HexDirection::Encode => "encode",
            HexDirection::Decode => "decode",
        };

        Ok(Some(
            TaskResult::new(
                task_id,
                pass,
                if pass { 1.0 } else { 0.0 },
                vec![direction.to_string()],
            )
            .with_metadata(Some(serde_json::json!({
                "plaintext": instance.plaintext,
                "hex_encoded": instance.hex_encoded,
                "direction": direction,
                "response": trimmed,
                "expected": expected,
            }))),
        ))
    }

    fn to_report_result(&self, b: &BenchmarkResult) -> Result<BenchmarkResult> {
        let (total, correct, output_tokens, thinking_tokens) = extract_task_stats(&b.raw);

        let accuracy = if total > 0 {
            correct as f64 / total as f64 * 100.0
        } else {
            0.0
        };

        let mut scores = BTreeMap::new();
        scores.insert(
            "accuracy".to_string(),
            Score::float(accuracy, ScoreUnit::Percent)
                .primary(true)
                .higher_is_better(true),
        );
        scores.insert("total".to_string(), Score::integer(total, ScoreUnit::Count));
        scores.insert(
            "correct".to_string(),
            Score::integer(correct, ScoreUnit::Count).higher_is_better(true),
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
                message: format!("Hex: {} correct out of {}", correct, total),
            }],
            raw: b.raw.clone(),
        })
    }
}

impl Benchmark for HexToolsBenchmark {
    fn name(&self) -> &str {
        "hex_tools"
    }

    fn display_name(&self) -> &'static str {
        "Hex Encoding/Decoding (Tools)"
    }

    fn category(&self) -> BenchmarkCategory {
        BenchmarkCategory::StringManipulation
    }

    fn pre_execute(&self, config: &yaml_serde::Value) -> Result<()> {
        load_config(&mut self.state.lock().unwrap(), config);
        Ok(())
    }

    fn execute_one(
        &self,
        model: &Model,
        _config: &yaml_serde::Value,
        tracker: &mut TokenTracker,
    ) -> Result<Option<TaskResult>> {
        let (instance, task_id) = {
            let mut state = self.state.lock().unwrap();
            if state.current_idx >= state.instances.len() {
                return Ok(None);
            }
            let instance = state.instances[state.current_idx].clone();
            let idx = state.current_idx;
            state.current_idx += 1;
            (instance, format!("task-{}", idx))
        };

        let system = "You have tools to hex encode and decode text. Use the appropriate tool to complete the task.";

        let tools: Vec<serde_json::Value> = vec![
            serde_json::json!({
                "type": "function",
                "function": {
                    "name": "hex_encode",
                    "description": "Encode text to hex string.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "text": {
                                "type": "string",
                                "description": "The text to encode"
                            }
                        },
                        "required": ["text"]
                    }
                }
            }),
            serde_json::json!({
                "type": "function",
                "function": {
                    "name": "hex_decode",
                    "description": "Decode hex string to text.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "hex_string": {
                                "type": "string",
                                "description": "The hex string to decode"
                            }
                        },
                        "required": ["hex_string"]
                    }
                }
            }),
        ];

        let (prompt, expected) = match instance.direction {
            HexDirection::Encode => (
                format!(
                    "Convert this text to hex encoding: '{}'",
                    instance.plaintext
                ),
                instance.hex_encoded.clone(),
            ),
            HexDirection::Decode => (
                format!(
                    "Convert this hex string to text: '{}'",
                    instance.hex_encoded
                ),
                instance.plaintext.clone(),
            ),
        };

        let (text, tool_calls) = tracker.chat_completion_with_tools(
            &model.model_name,
            system,
            &prompt,
            tools.clone(),
            None,
            false,
        )?;

        tracker.record_tool_calls(&tool_calls, &tools);

        let mut pass = false;
        let mut tool_name = String::new();
        let mut tool_input = String::new();

        if let Some(tc) = tool_calls.first() {
            tool_name = tc.name.clone();
            match tc.name.as_str() {
                "hex_encode" => {
                    if let Some(input) = tc.arguments.get("text").and_then(|v| v.as_str()) {
                        tool_input = input.to_string();
                        let encoded = hex_encode(input);
                        tracker.append_tool_result(&tc.id, &encoded);
                        pass = encoded == expected;
                    }
                }
                "hex_decode" => {
                    if let Some(input) = tc.arguments.get("hex_string").and_then(|v| v.as_str()) {
                        tool_input = input.to_string();
                        if let Ok(decoded) = hex_decode(input) {
                            tracker.append_tool_result(&tc.id, &decoded);
                            pass = decoded == expected;
                        }
                    }
                }
                _ => {}
            }
        } else {
            let trimmed = text.trim().to_lowercase();
            let normalized = trimmed
                .replace([' ', ':', '\n'], "")
                .strip_prefix("0x")
                .unwrap_or(&trimmed)
                .to_string();
            pass = normalized == expected;
        }

        let direction = match instance.direction {
            HexDirection::Encode => "encode",
            HexDirection::Decode => "decode",
        };

        Ok(Some(
            TaskResult::new(
                task_id,
                pass,
                if pass { 1.0 } else { 0.0 },
                vec![direction.to_string()],
            )
            .with_metadata(Some(serde_json::json!({
                "plaintext": instance.plaintext,
                "hex_encoded": instance.hex_encoded,
                "direction": direction,
                "response": text.trim(),
                "expected": expected,
                "tool_called": !tool_calls.is_empty(),
                "tool_name": tool_name,
                "tool_input": tool_input,
            }))),
        ))
    }

    fn to_report_result(&self, b: &BenchmarkResult) -> Result<BenchmarkResult> {
        let scores = b.scores.clone();

        Ok(BenchmarkResult {
            scores,
            breakdowns: BTreeMap::new(),
            error_classification: BTreeMap::new(),
            artifacts: vec![],
            diagnostics: vec![crate::reports::model::Diagnostic {
                level: "info".to_string(),
                message: format!(
                    "Hex (Tools): {} correct out of {}",
                    b.raw
                        .get("passed_tasks")
                        .and_then(|v| v.as_i64())
                        .unwrap_or(0),
                    b.raw
                        .get("total_tasks")
                        .and_then(|v| v.as_i64())
                        .unwrap_or(0),
                ),
            }],
            raw: b.raw.clone(),
        })
    }
}
