use crate::benchmarks::Benchmark;
use crate::config;
use crate::config::Model;
use crate::shared::{BenchmarkCategory, BenchmarkResult, TaskResult};
use crate::token_tracker::TokenTracker;
use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::sync::Mutex;

/// Reverse a string by Unicode scalar values (grapheme-agnostic, matching the
/// scoring logic in `execute_one`).
fn reverse_str(s: &str) -> String {
    s.chars().rev().collect()
}

/// Reverse writing benchmark: ask the model to reverse a word.
/// Direct prompt variant.
pub struct ReverseBenchmark {
    state: Mutex<ReverseState>,
}

/// Reverse writing benchmark: ask the model to reverse a word using a tool.
/// Tool-calling variant.
pub struct ReverseToolsBenchmark {
    state: Mutex<ReverseState>,
}

#[derive(Clone, Serialize, Deserialize)]
struct ReverseState {
    words: Vec<String>,
    current_idx: usize,
}

const DEFAULT_WORDS: &[&str] = &[
    "cat",
    "dog",
    "hello",
    "world",
    "test",
    "data",
    "reversed",
    "algorithm",
    "pelican",
    "moonwalk",
    "benchmark",
    "science",
    "machine",
    "learning",
    "language",
    "computer",
    "programming",
    "developer",
    "challenge",
    "difficulty",
    "function",
    "variable",
    "constant",
    "iterator",
    "generator",
    "callback",
    "promise",
    "recursive",
    "asynchronous",
    "polymorphic",
    "interface",
    "abstraction",
    "encapsulation",
    "inheritance",
    "composition",
    "dependency",
    "framework",
    "middleware",
    "singleton",
    "prototype",
    "observer",
    "factory",
    "strategy",
    "observer",
    "decorator",
    "adapter",
    "proxy",
    "facade",
    "bridge",
];

impl Default for ReverseBenchmark {
    fn default() -> Self {
        Self {
            state: Mutex::new(ReverseState {
                words: DEFAULT_WORDS.iter().map(|s| s.to_string()).collect(),
                current_idx: 0,
            }),
        }
    }
}

impl Default for ReverseToolsBenchmark {
    fn default() -> Self {
        Self {
            state: Mutex::new(ReverseState {
                words: DEFAULT_WORDS.iter().map(|s| s.to_string()).collect(),
                current_idx: 0,
            }),
        }
    }
}

impl Benchmark for ReverseBenchmark {
    fn name(&self) -> &str {
        "reverse"
    }

    fn display_name(&self) -> &'static str {
        "Reverse Writing"
    }

    fn category(&self) -> BenchmarkCategory {
        BenchmarkCategory::StringManipulation
    }

    fn pre_execute(&self, config: &yaml_serde::Value) -> Result<()> {
        let custom_words = config::extract_string_vec(config, "words");
        let num_samples = config::extract_usize(config, "num_samples");

        let mut state = self.state.lock().unwrap_or_else(|p| p.into_inner());
        if let Some(words) = custom_words {
            state.words = words;
        }
        if let Some(n) = num_samples {
            if n < state.words.len() {
                state.words.truncate(n);
            }
        }
        state.current_idx = 0;
        drop(state);

        Ok(())
    }

    fn execute_one(
        &self,
        model: &Model,
        _config: &yaml_serde::Value,
        tracker: &mut TokenTracker,
    ) -> Result<Option<TaskResult>> {
        let (word, task_id) = {
            let mut state = self.state.lock().unwrap_or_else(|p| p.into_inner());
            if state.current_idx >= state.words.len() {
                return Ok(None);
            }
            let word = state.words[state.current_idx].clone();
            let idx = state.current_idx;
            state.current_idx += 1;
            (word, format!("task-{}", idx))
        };

        let prompt = format!("Write the word '{}' in reverse.", word);
        let response = tracker.chat_completion(&model.model_name, "", &prompt)?;

        let expected = reverse_str(&word);
        let trimmed = response.trim();
        let pass = trimmed.to_lowercase() == expected.to_lowercase();

        Ok(Some(
            TaskResult::new(task_id, pass, if pass { 1.0 } else { 0.0 }, vec![]).with_metadata(
                Some(serde_json::json!({
                    "word": word,
                    "expected": expected,
                    "response": trimmed,
                })),
            ),
        ))
    }

    fn to_report_result(&self, b: &BenchmarkResult) -> Result<BenchmarkResult> {
        crate::utils::build_accuracy_result("Reverse", b)
    }
}

impl Benchmark for ReverseToolsBenchmark {
    fn name(&self) -> &str {
        "reverse_tools"
    }

    fn display_name(&self) -> &'static str {
        "Reverse Writing (Tools)"
    }

    fn category(&self) -> BenchmarkCategory {
        BenchmarkCategory::StringManipulation
    }

    fn pre_execute(&self, config: &yaml_serde::Value) -> Result<()> {
        let custom_words = config::extract_string_vec(config, "words");
        let num_samples = config::extract_usize(config, "num_samples");

        let mut state = self.state.lock().unwrap_or_else(|p| p.into_inner());
        if let Some(words) = custom_words {
            state.words = words;
        }
        if let Some(n) = num_samples {
            if n < state.words.len() {
                state.words.truncate(n);
            }
        }
        state.current_idx = 0;
        drop(state);

        Ok(())
    }

    fn execute_one(
        &self,
        model: &Model,
        _config: &yaml_serde::Value,
        tracker: &mut TokenTracker,
    ) -> Result<Option<TaskResult>> {
        let (word, task_id) = {
            let mut state = self.state.lock().unwrap_or_else(|p| p.into_inner());
            if state.current_idx >= state.words.len() {
                return Ok(None);
            }
            let word = state.words[state.current_idx].clone();
            let idx = state.current_idx;
            state.current_idx += 1;
            (word, format!("task-{}", idx))
        };

        let prompt = format!("Reverse the following word: '{}'", word);
        let system = "You have a tool to reverse strings. Use it to complete the task.";

        let tools: Vec<serde_json::Value> = vec![serde_json::json!({
            "type": "function",
            "function": {
                "name": "reverse_string",
                "description": "Reverse the given string character by character.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "input": {
                            "type": "string",
                            "description": "The string to reverse"
                        }
                    },
                    "required": ["input"]
                }
            }
        })];

        let (text, tool_calls) = tracker.chat_completion_with_tools(
            &model.model_name,
            system,
            &prompt,
            tools.clone(),
            None,
            false,
        )?;

        // Record tool calls for schema validation
        tracker.record_tool_calls(&tool_calls, &tools);

        // Simulate the tool locally: extract the input parameter and reverse it
        let expected = reverse_str(&word);
        let mut pass = false;
        let mut tool_input = String::new();

        if let Some(tc) = tool_calls.first() {
            if let Some(input) = tc.arguments.get("input").and_then(|v| v.as_str()) {
                tool_input = input.to_string();
                let reversed = input.chars().rev().collect::<String>();
                // Append the simulated tool result
                tracker.append_tool_result(&tc.id, &reversed);

                // Check if the final response matches the expected reversal
                let trimmed = text.trim();
                pass = trimmed.to_lowercase() == expected.to_lowercase()
                    || reversed.to_lowercase() == expected.to_lowercase();
            }
        } else {
            // No tool call — check if the text itself contains the answer
            let trimmed = text.trim();
            pass = trimmed.to_lowercase() == expected.to_lowercase();
        }

        Ok(Some(
            TaskResult::new(task_id, pass, if pass { 1.0 } else { 0.0 }, vec![]).with_metadata(
                Some(serde_json::json!({
                    "word": word,
                    "expected": expected,
                    "response": text.trim(),
                    "tool_called": !tool_calls.is_empty(),
                    "tool_input": tool_input,
                })),
            ),
        ))
    }

    fn to_report_result(&self, b: &BenchmarkResult) -> Result<BenchmarkResult> {
        let scores = b.scores.clone();
        // Add benchmark-specific diagnostic; pass_rate and tool_call_* are from b.scores

        Ok(BenchmarkResult {
            scores,
            breakdowns: BTreeMap::new(),
            error_classification: BTreeMap::new(),
            artifacts: vec![],
            diagnostics: vec![crate::reports::model::Diagnostic {
                level: "info".to_string(),
                message: format!(
                    "Reverse (Tools): {} correct out of {}",
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
    fn reverse_reverses_ascii() {
        assert_eq!(reverse_str("hello"), "olleh");
        assert_eq!(reverse_str("racecar"), "racecar");
        assert_eq!(reverse_str(""), "");
    }

    #[test]
    fn reverse_is_case_sensitive() {
        // Scoring compares case-insensitively, but the reversal itself is exact.
        assert_eq!(reverse_str("Hello"), "olleH");
    }

    #[test]
    fn reverse_handles_unicode_scalars() {
        assert_eq!(reverse_str("héllo"), "olléh");
    }
}
