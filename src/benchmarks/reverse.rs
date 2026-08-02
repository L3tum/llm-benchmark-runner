use crate::benchmarks::Benchmark;
use crate::config;
use crate::config::Model;
use crate::shared::{BenchmarkCategory, BenchmarkResult, Score, ScoreUnit, TaskResult};
use crate::token_tracker::TokenTracker;
use anyhow::Result;
use std::collections::BTreeMap;
use std::sync::Mutex;

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

        let mut state = self.state.lock().expect(crate::shared::MUTEX_PANIC_MSG);
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
            let mut state = self.state.lock().expect(crate::shared::MUTEX_PANIC_MSG);
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

        let expected = word.chars().rev().collect::<String>();
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
                message: format!("Reverse: {} correct out of {}", correct, total),
            }],
            raw: b.raw.clone(),
        })
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

        let mut state = self.state.lock().expect(crate::shared::MUTEX_PANIC_MSG);
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
            let mut state = self.state.lock().expect(crate::shared::MUTEX_PANIC_MSG);
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
        let expected = word.chars().rev().collect::<String>();
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

/// Extract common task stats from the raw JSON.
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
