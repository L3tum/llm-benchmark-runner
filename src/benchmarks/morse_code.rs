use crate::benchmarks::Benchmark;
use crate::config;
use crate::config::Model;
use crate::reports::model::{BenchmarkCategory, BenchmarkResult, Score, ScoreUnit, TaskResult};
use crate::token_tracker::TokenTracker;
use anyhow::Result;
use std::collections::BTreeMap;
use std::sync::Mutex;

/// Morse code translation benchmark: encode text to morse or decode morse to text.
/// Direct prompt variant.
pub struct MorseCodeBenchmark {
    state: Mutex<MorseCodeState>,
}

/// Morse code translation benchmark: encode/decode using tools.
/// Tool-calling variant.
pub struct MorseCodeToolsBenchmark {
    state: Mutex<MorseCodeState>,
}

struct MorseCodeState {
    instances: Vec<MorseInstance>,
    current_idx: usize,
}

#[derive(Clone)]
struct MorseInstance {
    text: String,
    morse: String,
    direction: MorseDirection,
}

#[derive(Clone)]
enum MorseDirection {
    Encode,
    Decode,
}

const DEFAULT_WORDS: &[&str] = &[
    "HELLO", "WORLD", "TEST", "DATA", "CODE", "BENCH", "MARK", "REVERSE", "MORSE", "SIGNAL",
    "MESSAGE", "ALPHA", "BRAVO", "CHARLIE", "DELTA", "ECHO", "FOXTROT", "GOLF", "HOTEL", "INDIA",
    "JULIET", "KILO", "LIMA", "MIKE", "NOVEMBER", "OSCAR", "PAPA", "QUEBEC", "ROMEO", "SIERRA",
    "TANGO", "UNIFORM", "VICTOR", "WHISKEY", "XRAY", "YANKEE", "ZULU", "PELICAN", "MOONWALK",
    "SCIENCE", "MACHINE", "LEARNING",
];

fn get_morse_dict() -> &'static std::collections::HashMap<char, &'static str> {
    use std::collections::HashMap;
    use std::sync::OnceLock;
    static DICT: OnceLock<HashMap<char, &str>> = OnceLock::new();
    DICT.get_or_init(|| {
        HashMap::from([
            ('A', ".-"),
            ('B', "-..."),
            ('C', "-.-."),
            ('D', "-.."),
            ('E', "."),
            ('F', "..-."),
            ('G', "--."),
            ('H', "...."),
            ('I', ".."),
            ('J', ".---"),
            ('K', "-.-"),
            ('L', ".-.."),
            ('M', "--"),
            ('N', "-."),
            ('O', "---"),
            ('P', ".--."),
            ('Q', "--.-"),
            ('R', ".-."),
            ('S', "..."),
            ('T', "-"),
            ('U', "..-"),
            ('V', "...-"),
            ('W', ".--"),
            ('X', "-..-"),
            ('Y', "-.--"),
            ('Z', "--.."),
            ('0', "-----"),
            ('1', ".----"),
            ('2', "..---"),
            ('3', "...--"),
            ('4', "....-"),
            ('5', "....."),
            ('6', "-...."),
            ('7', "--..."),
            ('8', "---.."),
            ('9', "----."),
            (' ', "/"),
        ])
    })
}

impl Default for MorseCodeBenchmark {
    fn default() -> Self {
        Self {
            state: Mutex::new(MorseCodeState {
                instances: Vec::new(),
                current_idx: 0,
            }),
        }
    }
}

impl Default for MorseCodeToolsBenchmark {
    fn default() -> Self {
        Self {
            state: Mutex::new(MorseCodeState {
                instances: Vec::new(),
                current_idx: 0,
            }),
        }
    }
}

fn text_to_morse(text: &str) -> String {
    let dict = get_morse_dict();
    text.chars()
        .map(|c| {
            dict.get(&c.to_uppercase().next().unwrap_or(c))
                .copied()
                .unwrap_or("?")
        })
        .collect::<Vec<&str>>()
        .join(" ")
}

fn morse_to_text(morse: &str) -> String {
    let dict = get_morse_dict();
    // Build reverse lookup: morse_code -> char
    let reverse: std::collections::HashMap<&str, char> =
        dict.iter().map(|(&ch, &code)| (code, ch)).collect();
    morse
        .split('/')
        .map(|word| {
            word.split_whitespace()
                .map(|code| reverse.get(code).copied().unwrap_or('?'))
                .collect::<String>()
        })
        .collect::<Vec<String>>()
        .join(" ")
}

fn normalize_morse(s: &str) -> String {
    s.split_whitespace()
        .collect::<Vec<&str>>()
        .join(" ")
        .to_uppercase()
}

fn build_instances(word_list: &[String]) -> Vec<MorseInstance> {
    let mut instances = Vec::new();
    for word in word_list {
        let morse = text_to_morse(word);
        instances.push(MorseInstance {
            text: word.to_uppercase(),
            morse: morse.clone(),
            direction: MorseDirection::Encode,
        });
        instances.push(MorseInstance {
            text: word.to_uppercase(),
            morse,
            direction: MorseDirection::Decode,
        });
    }
    instances
}

fn load_config(state: &mut MorseCodeState, config: &yaml_serde::Value) {
    let custom_words = config::extract_string_vec(config, "text_samples");
    let num_samples = config::extract_usize(config, "num_samples");

    let word_list =
        custom_words.unwrap_or_else(|| DEFAULT_WORDS.iter().map(|s| s.to_string()).collect());

    state.instances = build_instances(&word_list);

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

impl Benchmark for MorseCodeBenchmark {
    fn name(&self) -> &str {
        "morse_code"
    }

    fn display_name(&self) -> &'static str {
        "Morse Code Translation"
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
            MorseDirection::Encode => (
                format!(
                    "Convert this text to Morse code: '{}'\nReturn only the Morse code.",
                    instance.text
                ),
                instance.morse.clone(),
            ),
            MorseDirection::Decode => (
                format!(
                    "Convert this Morse code to text: '{}'\nReturn only the text.",
                    instance.morse
                ),
                instance.text.clone(),
            ),
        };

        let response = tracker.chat_completion(&model.model_name, "", &prompt)?;
        let trimmed = response.trim();

        let normalized_response = normalize_morse(trimmed);
        let normalized_expected = normalize_morse(&expected);
        let pass = normalized_response == normalized_expected;

        let direction = match instance.direction {
            MorseDirection::Encode => "encode",
            MorseDirection::Decode => "decode",
        };

        Ok(Some(
            TaskResult::new(
                task_id,
                pass,
                if pass { 1.0 } else { 0.0 },
                vec![direction.to_string()],
            )
            .with_metadata(Some(serde_json::json!({
                "text": instance.text,
                "morse": instance.morse,
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
                message: format!("Morse Code: {} correct out of {}", correct, total),
            }],
            raw: b.raw.clone(),
        })
    }
}

impl Benchmark for MorseCodeToolsBenchmark {
    fn name(&self) -> &str {
        "morse_code_tools"
    }

    fn display_name(&self) -> &'static str {
        "Morse Code Translation (Tools)"
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

        let system = "You have tools to encode text to Morse code and decode Morse code to text. Use the appropriate tool to complete the task.";

        let tools: Vec<serde_json::Value> = vec![
            serde_json::json!({
                "type": "function",
                "function": {
                    "name": "encode_morse",
                    "description": "Encode text to Morse code.",
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
                    "name": "decode_morse",
                    "description": "Decode Morse code to text.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "morse": {
                                "type": "string",
                                "description": "The Morse code to decode"
                            }
                        },
                        "required": ["morse"]
                    }
                }
            }),
        ];

        let (prompt, expected) = match instance.direction {
            MorseDirection::Encode => (
                format!("Convert this text to Morse code: '{}'", instance.text),
                instance.morse.clone(),
            ),
            MorseDirection::Decode => (
                format!("Convert this Morse code to text: '{}'", instance.morse),
                instance.text.clone(),
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

        let expected_normalized = normalize_morse(&expected);
        let mut pass = false;
        let mut tool_name = String::new();
        let mut tool_input = String::new();

        if let Some(tc) = tool_calls.first() {
            tool_name = tc.name.clone();
            match tc.name.as_str() {
                "encode_morse" => {
                    if let Some(input) = tc.arguments.get("text").and_then(|v| v.as_str()) {
                        tool_input = input.to_string();
                        let encoded = text_to_morse(input);
                        tracker.append_tool_result(&tc.id, &encoded);
                        pass = normalize_morse(&encoded) == expected_normalized;
                    }
                }
                "decode_morse" => {
                    if let Some(input) = tc.arguments.get("morse").and_then(|v| v.as_str()) {
                        tool_input = input.to_string();
                        let decoded = morse_to_text(input);
                        tracker.append_tool_result(&tc.id, &decoded);
                        pass = normalize_morse(&decoded) == expected_normalized;
                    }
                }
                _ => {}
            }
        } else {
            let trimmed = text.trim();
            pass = normalize_morse(trimmed) == expected_normalized;
        }

        let direction = match instance.direction {
            MorseDirection::Encode => "encode",
            MorseDirection::Decode => "decode",
        };

        Ok(Some(
            TaskResult::new(
                task_id,
                pass,
                if pass { 1.0 } else { 0.0 },
                vec![direction.to_string()],
            )
            .with_metadata(Some(serde_json::json!({
                "text": instance.text,
                "morse": instance.morse,
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
                    "Morse Code (Tools): {} correct out of {}",
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
