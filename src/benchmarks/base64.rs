use crate::benchmarks::Benchmark;
use crate::config;
use crate::config::Model;
use crate::shared::{BenchmarkCategory, BenchmarkResult, TaskResult};
use crate::token_tracker::TokenTracker;
use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::sync::Mutex;

/// Base64 encoding/decoding benchmark.
/// Direct prompt variant.
pub struct Base64Benchmark {
    state: Mutex<Base64State>,
}

/// Base64 encoding/decoding benchmark using tools.
/// Tool-calling variant.
pub struct Base64ToolsBenchmark {
    state: Mutex<Base64State>,
}

#[derive(Clone, Serialize, Deserialize)]
struct Base64State {
    instances: Vec<Base64Instance>,
    current_idx: usize,
}

#[derive(Clone, Serialize, Deserialize)]
struct Base64Instance {
    plaintext: String,
    encoded: String,
    direction: Base64Direction,
}

#[derive(Clone, PartialEq, Debug, Serialize, Deserialize)]
enum Base64Direction {
    Encode,
    Decode,
}

const DEFAULT_PLAINTEXTS: &[&str] = &[
    "hello",
    "world",
    "test data",
    "base64 encoding",
    "benchmark",
    "pelican",
    "moonwalk",
    "science machine",
    "learning language",
    "computer programming",
    "developer challenge",
    "function variable",
    "constant iterator",
    "generator callback",
    "promise recursive",
    "asynchronous polymorphic",
    "interface abstraction",
    "encapsulation inheritance",
    "composition dependency",
    "framework middleware",
    "singleton prototype",
    "observer factory",
    "strategy decorator",
    "adapter proxy facade",
    "bridge pattern",
];

impl Default for Base64Benchmark {
    fn default() -> Self {
        Self {
            state: Mutex::new(Base64State {
                instances: Vec::new(),
                current_idx: 0,
            }),
        }
    }
}

impl Default for Base64ToolsBenchmark {
    fn default() -> Self {
        Self {
            state: Mutex::new(Base64State {
                instances: Vec::new(),
                current_idx: 0,
            }),
        }
    }
}

fn base64_encode(input: &str) -> String {
    use base64::Engine;
    base64::engine::general_purpose::STANDARD.encode(input.as_bytes())
}

fn base64_decode(input: &str) -> Result<String> {
    use base64::Engine;
    let bytes = base64::engine::general_purpose::STANDARD.decode(input)?;
    String::from_utf8(bytes).map_err(|e| anyhow::anyhow!(e))
}

fn build_instances(word_list: &[String]) -> Vec<Base64Instance> {
    let mut instances = Vec::new();
    for text in word_list {
        let encoded = base64_encode(text);
        instances.push(Base64Instance {
            plaintext: text.clone(),
            encoded: encoded.clone(),
            direction: Base64Direction::Encode,
        });
        instances.push(Base64Instance {
            plaintext: text.clone(),
            encoded,
            direction: Base64Direction::Decode,
        });
    }
    instances
}

fn load_config(state: &mut Base64State, config: &yaml_serde::Value) {
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

impl Benchmark for Base64Benchmark {
    fn name(&self) -> &str {
        "base64"
    }

    fn display_name(&self) -> &'static str {
        "Base64 Encoding/Decoding"
    }

    fn category(&self) -> BenchmarkCategory {
        BenchmarkCategory::StringManipulation
    }

    fn pre_execute(&self, config: &yaml_serde::Value) -> Result<()> {
        load_config(
            &mut self.state.lock().unwrap_or_else(|p| p.into_inner()),
            config,
        );
        Ok(())
    }

    fn execute_one(
        &self,
        model: &Model,
        _config: &yaml_serde::Value,
        tracker: &mut TokenTracker,
    ) -> Result<Option<TaskResult>> {
        let (instance, task_id) = {
            let mut state = self.state.lock().unwrap_or_else(|p| p.into_inner());
            if state.current_idx >= state.instances.len() {
                return Ok(None);
            }
            let instance = state.instances[state.current_idx].clone();
            let idx = state.current_idx;
            state.current_idx += 1;
            (instance, format!("task-{}", idx))
        };

        let (prompt, expected) = match instance.direction {
            Base64Direction::Encode => (
                format!(
                    "Base64 encode this text: '{}'\nReturn only the base64 encoded string.",
                    instance.plaintext
                ),
                instance.encoded.clone(),
            ),
            Base64Direction::Decode => (
                format!(
                    "Base64 decode this string: '{}'\nReturn only the decoded text.",
                    instance.encoded
                ),
                instance.plaintext.clone(),
            ),
        };

        let response = tracker.chat_completion(&model.model_name, "", &prompt)?;
        let trimmed = response.trim();

        let pass = trimmed == expected;

        let direction = match instance.direction {
            Base64Direction::Encode => "encode",
            Base64Direction::Decode => "decode",
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
                "encoded": instance.encoded,
                "direction": direction,
                "response": trimmed,
                "expected": expected,
            }))),
        ))
    }

    fn to_report_result(&self, b: &BenchmarkResult) -> Result<BenchmarkResult> {
        crate::utils::build_accuracy_result("Base64", b)
    }
}

impl Benchmark for Base64ToolsBenchmark {
    fn name(&self) -> &str {
        "base64_tools"
    }

    fn display_name(&self) -> &'static str {
        "Base64 Encoding/Decoding (Tools)"
    }

    fn category(&self) -> BenchmarkCategory {
        BenchmarkCategory::StringManipulation
    }

    fn pre_execute(&self, config: &yaml_serde::Value) -> Result<()> {
        load_config(
            &mut self.state.lock().unwrap_or_else(|p| p.into_inner()),
            config,
        );
        Ok(())
    }

    fn execute_one(
        &self,
        model: &Model,
        _config: &yaml_serde::Value,
        tracker: &mut TokenTracker,
    ) -> Result<Option<TaskResult>> {
        let (instance, task_id) = {
            let mut state = self.state.lock().unwrap_or_else(|p| p.into_inner());
            if state.current_idx >= state.instances.len() {
                return Ok(None);
            }
            let instance = state.instances[state.current_idx].clone();
            let idx = state.current_idx;
            state.current_idx += 1;
            (instance, format!("task-{}", idx))
        };

        let system = "You have tools to base64 encode and decode text. Use the appropriate tool to complete the task.";

        let tools: Vec<serde_json::Value> = vec![
            serde_json::json!({
                "type": "function",
                "function": {
                    "name": "base64_encode",
                    "description": "Encode text to base64.",
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
                    "name": "base64_decode",
                    "description": "Decode base64 to text.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "encoded": {
                                "type": "string",
                                "description": "The base64 string to decode"
                            }
                        },
                        "required": ["encoded"]
                    }
                }
            }),
        ];

        let (prompt, expected) = match instance.direction {
            Base64Direction::Encode => (
                format!("Base64 encode this text: '{}'", instance.plaintext),
                instance.encoded.clone(),
            ),
            Base64Direction::Decode => (
                format!("Base64 decode this string: '{}'", instance.encoded),
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
                "base64_encode" => {
                    if let Some(input) = tc.arguments.get("text").and_then(|v| v.as_str()) {
                        tool_input = input.to_string();
                        let encoded = base64_encode(input);
                        tracker.append_tool_result(&tc.id, &encoded);
                        pass = encoded == expected;
                    }
                }
                "base64_decode" => {
                    if let Some(input) = tc.arguments.get("encoded").and_then(|v| v.as_str()) {
                        tool_input = input.to_string();
                        if let Ok(decoded) = base64_decode(input) {
                            tracker.append_tool_result(&tc.id, &decoded);
                            pass = decoded == expected;
                        }
                    }
                }
                _ => {}
            }
        } else {
            let trimmed = text.trim();
            pass = trimmed == expected;
        }

        let direction = match instance.direction {
            Base64Direction::Encode => "encode",
            Base64Direction::Decode => "decode",
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
                "encoded": instance.encoded,
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
                    "Base64 (Tools): {} correct out of {}",
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn base64_encode_round_trip() {
        assert_eq!(base64_encode("hello"), "aGVsbG8=");
        assert_eq!(base64_encode(""), "");
        assert_eq!(base64_encode("base64 works"), "YmFzZTY0IHdvcmtz");
    }

    #[test]
    fn base64_decode_round_trip() {
        assert_eq!(base64_decode("aGVsbG8=").unwrap(), "hello");
        assert_eq!(base64_decode("").unwrap(), "");
        assert_eq!(base64_decode("YmFzZTY0IHdvcmtz").unwrap(), "base64 works");
    }

    #[test]
    fn base64_decode_rejects_invalid_input() {
        assert!(base64_decode("###not valid base64###").is_err());
    }

    #[test]
    fn build_instances_creates_encode_and_decode_pairs() {
        let words = vec!["hi".to_string(), "there".to_string()];
        let instances = build_instances(&words);
        assert_eq!(instances.len(), 4);
        // Every plaintext appears once as Encode and once as Decode.
        assert_eq!(
            instances
                .iter()
                .filter(|i| i.plaintext == "hi" && i.direction == Base64Direction::Encode)
                .count(),
            1
        );
        assert_eq!(
            instances
                .iter()
                .filter(|i| i.plaintext == "hi" && i.direction == Base64Direction::Decode)
                .count(),
            1
        );
        // Encode instances' expected output is the base64 of the plaintext.
        let encode_hi = instances
            .iter()
            .find(|i| i.plaintext == "hi" && i.direction == Base64Direction::Encode)
            .unwrap();
        assert_eq!(encode_hi.encoded, base64_encode("hi"));
    }

    #[test]
    fn state_serializes_and_round_trips() {
        // C6: state must be (de)serializable to support per-task resume.
        let state = Base64State {
            instances: build_instances(&["a".to_string(), "b".to_string()]),
            current_idx: 3,
        };
        let json = serde_json::to_string(&state).expect("serialize");
        let back: Base64State = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(back.current_idx, 3);
        assert_eq!(back.instances.len(), state.instances.len());
        assert_eq!(back.instances[0].plaintext, "a");
        assert_eq!(back.instances[1].direction, Base64Direction::Decode);
    }
}
