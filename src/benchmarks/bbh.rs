use crate::benchmarks::Benchmark;
use crate::config::Model;
use crate::download::download_parquet_records;
use crate::reports::model::{BreakdownTable, Diagnostic};
use crate::shared::{BenchmarkCategory, BenchmarkResult, Score, ScoreUnit, TaskResult};
use crate::token_tracker::TokenTracker;
use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::fs;
use std::sync::Mutex;

pub struct BbhBenchmark {
    state: Mutex<BbhState>,
}

#[derive(Clone, serde::Serialize, serde::Deserialize)]
struct BbhState {
    tasks: Vec<BbhTask>,
    current_idx: usize,
}

impl Default for BbhBenchmark {
    fn default() -> Self {
        Self {
            state: Mutex::new(BbhState {
                tasks: Vec::new(),
                current_idx: 0,
            }),
        }
    }
}

#[derive(Debug, Clone, Deserialize, Serialize)]
struct BbhTask {
    #[serde(rename = "input")]
    input_text: String,
    target: String,
    task_name: String,
    #[serde(default)]
    few_shot_examples: Vec<BbhExample>,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
struct BbhExample {
    #[serde(rename = "input")]
    input_text: String,
    target: String,
}

/// BBH task definitions — focus on the contradiction/logic tasks most relevant to the plan.
const BBH_TASKS: &[&str] = &[
    "logical_deduction_three_objects",
    "logical_deduction_five_objects",
    "temporal_sequences",
    "disambiguation_qa",
    "hyperbaton",
    "reasoning_about_colored_objects",
    "object_counting",
    "tracking_shuffled_objects_three_objects",
    "tracking_shuffled_objects_five_objects",
    "tracking_shuffled_objects_seven_objects",
    "date_understanding",
    "navigate",
    "penguins_in_a_table",
    "reasoning_about_color",
    "boolean_expressions",
    "movie_recommendation",
    "salient_translation_error_detection",
    "cause_and_effect",
    "multistep_arithmetic_two",
    "web_of_lies",
    "formal_fallacies",
    "sports_understanding",
    "dyck_languages",
];

/// Task-specific instructions for BBH
fn task_instruction(task_name: &str) -> &'static str {
    match task_name {
        "logical_deduction_three_objects" => {
            "You are given three objects and a set of clues. Deduce the correct ordering."
        }
        "logical_deduction_five_objects" => {
            "You are given five objects and a set of clues. Deduce the correct ordering."
        }
        "temporal_sequences" => "Given a sequence of events, determine the correct temporal order.",
        "disambiguation_qa" => {
            "Given a passage with ambiguous references, answer questions about what each reference means."
        }
        "hyperbaton" => {
            "Determine whether a sentence with a syntactic manipulation preserves its meaning."
        }
        "reasoning_about_colored_objects" | "reasoning_about_color" => {
            "Reason about objects with different colors and their properties."
        }
        "object_counting" => "Count objects described in a text.",
        "tracking_shuffled_objects_three_objects" => {
            "Track the positions of three objects as they are shuffled around."
        }
        "tracking_shuffled_objects_five_objects" => {
            "Track the positions of five objects as they are shuffled around."
        }
        "tracking_shuffled_objects_seven_objects" => {
            "Track the positions of seven objects as they are shuffled around."
        }
        "date_understanding" => "Understand and reason about dates and temporal relationships.",
        "navigate" => "Follow directions and determine the final position.",
        "penguins_in_a_table" => "Answer questions based on a table of penguin data.",
        "boolean_expressions" => "Evaluate boolean expressions.",
        "movie_recommendation" => {
            "Given a movie description, recommend similar movies from a list."
        }
        "salient_translation_error_detection" => {
            "Detect the most salient error in a translated text."
        }
        "cause_and_effect" => "Identify cause and effect relationships in short scenarios.",
        "multistep_arithmetic_two" => "Solve multi-step arithmetic problems.",
        "web_of_lies" => "Determine the truth value of statements based on who said what.",
        "formal_fallacies" => "Identify whether arguments are valid or fallacious.",
        "sports_understanding" => "Answer questions based on sports rules and scenarios.",
        "dyck_languages" => "Complete a Dyck language string (balanced parentheses).",
        _ => "Answer the following question based on the provided information.",
    }
}

/// Convert BbhInstance objects into BbhTask objects for a given task name.
fn instances_to_tasks(instances: Vec<BbhInstance>, task_name: &str) -> Vec<BbhTask> {
    instances
        .into_iter()
        .map(|item| BbhTask {
            input_text: item.input,
            target: item.target,
            task_name: task_name.to_string(),
            few_shot_examples: item
                .few_shot_examples
                .unwrap_or_default()
                .into_iter()
                .map(|e| BbhExample {
                    input_text: e.input,
                    target: e.target,
                })
                .collect(),
        })
        .collect()
}

fn load_bbh_tasks(selected_tasks: &[&str]) -> Vec<BbhTask> {
    let cache_dir = dirs::cache_dir()
        .unwrap_or_default()
        .join("llm-benchmark-runner")
        .join("bbh");

    fs::create_dir_all(&cache_dir).expect("Failed to create cache dir");

    let mut all_tasks = Vec::new();

    for task_name in selected_tasks {
        let task_file = cache_dir.join(format!("{}.json", task_name));

        let instances: Vec<BbhInstance> = if task_file.exists() {
            let content = fs::read_to_string(&task_file).expect("Failed to read cached BBH task");
            serde_json::from_str(&content).expect("Failed to parse BBH task")
        } else {
            // BBH on HuggingFace is hosted as per-task parquet shards.
            let url = format!(
                "https://huggingface.co/datasets/lukaemon/bbh/resolve/main/{}/test-00000-of-00001.parquet",
                task_name
            );
            let rows = download_parquet_records(&url, 2, 120, "llm-benchmark-runner")
                .unwrap_or_else(|e| {
                    eprintln!("  Failed to download BBH task {}: {}", task_name, e);
                    Vec::new()
                });
            let inst: Vec<BbhInstance> = rows
                .iter()
                .map(|r| BbhInstance {
                    input: r
                        .get("input")
                        .and_then(|v| v.as_str())
                        .unwrap_or("")
                        .to_string(),
                    target: r
                        .get("target")
                        .and_then(|v| v.as_str())
                        .unwrap_or("")
                        .to_string(),
                    few_shot_examples: None,
                })
                .collect();
            fs::write(
                &task_file,
                serde_json::to_vec(&inst).expect("Failed to save BBH task"),
            )
            .expect("Failed to save BBH task");
            inst
        };
        all_tasks.extend(instances_to_tasks(instances, task_name));
    }

    all_tasks
}

#[derive(Debug, Serialize, Deserialize)]
struct BbhInstance {
    input: String,
    target: String,
    #[serde(default)]
    few_shot_examples: Option<Vec<BbhRawExample>>,
}

#[derive(Debug, Serialize, Deserialize)]
struct BbhRawExample {
    input: String,
    target: String,
}

impl Benchmark for BbhBenchmark {
    fn name(&self) -> &str {
        "bbh"
    }

    fn display_name(&self) -> &'static str {
        "BBH (Big-Bench Hard — Logic & Reasoning)"
    }

    fn category(&self) -> BenchmarkCategory {
        BenchmarkCategory::Reasoning
    }

    fn pre_execute(&self, config: &yaml_serde::Value) -> Result<()> {
        // Default to logic/contradiction tasks; can be overridden
        let selected_tasks =
            if let Some(tasks_val) = config.get("tasks").and_then(|v| v.as_sequence()) {
                tasks_val
                    .iter()
                    .filter_map(|t| t.as_str())
                    .collect::<Vec<&str>>()
            } else {
                // Default: use first 8 tasks from BBH_TASKS
                BBH_TASKS[..8].to_vec()
            };

        let tasks = load_bbh_tasks(&selected_tasks);
        println!(
            "  BBH: {} tasks from {} task types loaded",
            tasks.len(),
            selected_tasks.len()
        );
        let mut state = self.state.lock().unwrap_or_else(|p| p.into_inner());
        state.tasks = tasks;
        state.current_idx = 0;
        Ok(())
    }

    fn execute_one(
        &self,
        model: &Model,
        _config: &yaml_serde::Value,
        tracker: &mut TokenTracker,
    ) -> Result<Option<TaskResult>> {
        let (task, idx) = {
            let mut state = self.state.lock().unwrap_or_else(|p| p.into_inner());
            if state.current_idx >= state.tasks.len() {
                return Ok(None);
            }
            let idx = state.current_idx;
            let task = state.tasks[idx].clone();
            state.current_idx += 1;
            (task, idx)
        };

        let system_prompt = task_instruction(&task.task_name);

        // Build few-shot examples
        let examples_str = task
            .few_shot_examples
            .iter()
            .map(|e| format!("Input: {}\nAnswer: {}\n", e.input_text, e.target))
            .collect::<Vec<_>>()
            .join("\n");

        let user_prompt = format!(
            "{}\n\n{}Input: {}\nAnswer:",
            system_prompt,
            if examples_str.is_empty() {
                String::new()
            } else {
                format!("Examples:\n\n{}", examples_str)
            },
            task.input_text
        );

        let response = tracker.chat_completion(&model.model_name, system_prompt, &user_prompt)?;
        let response_trimmed = response.trim();

        // Exact match or substring containment for BBH
        let target_trimmed = task.target.trim();
        let is_correct = response_trimmed == target_trimmed
            || response_trimmed.to_lowercase() == target_trimmed.to_lowercase()
            || (target_trimmed.len() < 10
                && response_trimmed
                    .to_lowercase()
                    .contains(&target_trimmed.to_lowercase()));

        Ok(Some(
            TaskResult::new(
                format!("task-{}", idx),
                is_correct,
                if is_correct { 1.0 } else { 0.0 },
                vec![task.task_name.clone()],
            )
            .with_metadata(Some(serde_json::json!({
                "task_name": task.task_name,
                "expected": task.target,
                "response": response_trimmed,
            }))),
        ))
    }

    fn to_report_result(&self, b: &BenchmarkResult) -> Result<BenchmarkResult> {
        let raw = &b.raw;

        let (total, correct, output_tokens, thinking_tokens, task_stats) = {
            if let Some(per_task) = raw.get("per_task").and_then(|v| v.as_array()) {
                let total = per_task.len() as i64;
                let correct = per_task
                    .iter()
                    .filter(|t| t.get("passed").and_then(|v| v.as_bool()).unwrap_or(false))
                    .count() as i64;
                let out: i64 = per_task
                    .iter()
                    .filter_map(|t| t.get("output_tokens").and_then(|v| v.as_i64()))
                    .sum();
                let think: i64 = per_task
                    .iter()
                    .filter_map(|t| t.get("thinking_tokens").and_then(|v| v.as_i64()))
                    .sum();

                let mut task_stats: BTreeMap<String, (i64, i64)> = BTreeMap::new();
                for task in per_task {
                    if let Some(task_name) = task
                        .get("categories")
                        .and_then(|v| v.as_array())
                        .and_then(|arr| arr.first())
                        .and_then(|v| v.as_str())
                    {
                        let (p, t) = task_stats.entry(task_name.to_string()).or_insert((0, 0));
                        *t += 1;
                        if task
                            .get("passed")
                            .and_then(|v| v.as_bool())
                            .unwrap_or(false)
                        {
                            *p += 1;
                        }
                    }
                }
                (total, correct, out, think, task_stats)
            } else {
                (
                    raw.get("total").and_then(|v| v.as_i64()).unwrap_or(0),
                    raw.get("correct").and_then(|v| v.as_i64()).unwrap_or(0),
                    raw.get("output_tokens")
                        .and_then(|v| v.as_i64())
                        .unwrap_or(0),
                    raw.get("thinking_tokens")
                        .and_then(|v| v.as_i64())
                        .unwrap_or(0),
                    BTreeMap::new(),
                )
            }
        };

        let accuracy = if total > 0 {
            correct as f64 / total as f64
        } else {
            0.0
        };

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

        // Per-task breakdown
        let mut breakdowns = BTreeMap::new();
        if !task_stats.is_empty() {
            let mut rows = BTreeMap::new();
            for (task_name, (task_correct, task_total)) in &task_stats {
                let rate = if *task_total > 0 {
                    *task_correct as f64 / *task_total as f64
                } else {
                    0.0
                };
                rows.insert(
                    task_name.clone(),
                    BTreeMap::from([
                        (
                            "accuracy".to_string(),
                            Score::float(rate * 100.0, ScoreUnit::Percent)
                                .display(format!("{:.1}%", rate * 100.0)),
                        ),
                        (
                            "instances".to_string(),
                            Score::integer(*task_total, ScoreUnit::Count)
                                .display(format!("{}/{}", task_correct, task_total)),
                        ),
                    ]),
                );
            }
            breakdowns.insert(
                "By Task".to_string(),
                BreakdownTable {
                    title: "Accuracy by BBH Task".to_string(),
                    rows,
                },
            );
        }

        Ok(BenchmarkResult {
            scores,
            breakdowns,
            error_classification: BTreeMap::new(),
            artifacts: vec![],
            diagnostics: vec![Diagnostic {
                level: "info".to_string(),
                message: format!(
                    "BBH: {}/{} correct ({:.1}%) across {} task types",
                    correct,
                    total,
                    accuracy * 100.0,
                    task_stats.len()
                ),
            }],
            raw: raw.clone(),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_task_instruction_known() {
        let instr = task_instruction("date_understanding");
        assert!(!instr.is_empty());
    }

    #[test]
    fn test_task_instruction_unknown() {
        let instr = task_instruction("nonexistent_task_xyz");
        assert_eq!(
            instr,
            "Answer the following question based on the provided information."
        );
    }

    #[test]
    fn test_instances_to_tasks_empty() {
        let tasks = instances_to_tasks(Vec::new(), "test_task");
        assert!(tasks.is_empty());
    }

    #[test]
    fn test_instances_to_tasks_populated() {
        let instances = vec![BbhInstance {
            input: "What is 2+2?".to_string(),
            target: "4".to_string(),
            few_shot_examples: None,
        }];
        let tasks = instances_to_tasks(instances, "simple_math");
        assert_eq!(tasks.len(), 1);
        assert_eq!(tasks[0].input_text, "What is 2+2?");
    }
}
