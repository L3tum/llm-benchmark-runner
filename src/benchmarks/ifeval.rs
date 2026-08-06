use crate::benchmarks::Benchmark;
use crate::config::Model;
use crate::download::download_with_retry_bytes;
use crate::shared::{BenchmarkCategory, BenchmarkResult, Score, ScoreUnit, TaskResult};
use crate::token_tracker::TokenTracker;
use anyhow::{Context, Result};
use serde::Deserialize;
use std::collections::BTreeMap;
use std::fs;
use std::sync::{Arc, Mutex};

pub struct IFEvalBenchmark {
    state: Mutex<IFEvalState>,
}

#[derive(Clone, serde::Serialize, serde::Deserialize)]
struct IFEvalState {
    items: Vec<IFEvalRow>,
    current_idx: usize,
}

impl Default for IFEvalBenchmark {
    fn default() -> Self {
        Self {
            state: Mutex::new(IFEvalState {
                items: Vec::new(),
                current_idx: 0,
            }),
        }
    }
}

const IFEVAL_URL: &str =
    "https://huggingface.co/datasets/google/IFEval/resolve/main/ifeval_input_data.jsonl";

#[derive(Debug, Clone, Deserialize, serde::Serialize)]
struct IFEvalRow {
    prompt: String,
    #[serde(rename = "instruction_id_list")]
    instruction_ids: Vec<String>,
}

struct InstructionVerifier {
    id: String,
    check_fn: Arc<dyn Fn(&str) -> bool + Send + Sync>,
}

impl Clone for InstructionVerifier {
    fn clone(&self) -> Self {
        Self {
            id: self.id.clone(),
            check_fn: Arc::clone(&self.check_fn),
        }
    }
}

fn load_ifeval_dataset() -> Result<Vec<IFEvalRow>> {
    let cache_dir = dirs::cache_dir()
        .unwrap_or_default()
        .join("llm-benchmark-runner")
        .join("ifeval");
    let path = cache_dir.join("ifeval_input_data.jsonl");

    if path.exists() {
        return parse_ifeval_jsonl(&fs::read_to_string(&path)?);
    }

    fs::create_dir_all(&cache_dir)?;
    println!("  Downloading IFEval dataset...");
    let bytes = download_with_retry_bytes(IFEVAL_URL, 3, 60, "llm-benchmark-runner")?;
    let tmp_path = path.with_extension(format!("jsonl.tmp.{}", std::process::id()));
    fs::write(&tmp_path, &bytes)?;
    fs::rename(&tmp_path, &path).context("failed to rename IFEval download")?;
    parse_ifeval_jsonl(&String::from_utf8_lossy(&bytes))
}

/// Parse a JSONL document (one JSON object per line) into rows.
fn parse_ifeval_jsonl(content: &str) -> Result<Vec<IFEvalRow>> {
    content
        .lines()
        .map(|line| -> Result<IFEvalRow> { Ok(serde_json::from_str(line)?) })
        .collect()
}

fn create_verifiers(instruction_ids: &[String]) -> Vec<InstructionVerifier> {
    let mut verifiers = Vec::new();

    for id in instruction_ids {
        if let Some(verifier) = create_single_verifier(id) {
            verifiers.push(verifier);
        }
    }

    verifiers
}

fn create_single_verifier(id: &str) -> Option<InstructionVerifier> {
    match id {
        // Length constraints
        "word_count_more_than_20" => Some(InstructionVerifier {
            id: id.to_string(),
            check_fn: Arc::new(|text| word_count(text) > 20),
        }),
        "word_count_more_than_50" => Some(InstructionVerifier {
            id: id.to_string(),
            check_fn: Arc::new(|text| word_count(text) > 50),
        }),
        "word_count_more_than_100" => Some(InstructionVerifier {
            id: id.to_string(),
            check_fn: Arc::new(|text| word_count(text) > 100),
        }),
        "word_count_less_than_20" => Some(InstructionVerifier {
            id: id.to_string(),
            check_fn: Arc::new(|text| word_count(text) <= 20),
        }),
        "word_count_less_than_50" => Some(InstructionVerifier {
            id: id.to_string(),
            check_fn: Arc::new(|text| word_count(text) <= 50),
        }),
        "word_count_equal_to_20" => Some(InstructionVerifier {
            id: id.to_string(),
            check_fn: Arc::new(|text| word_count(text) == 20),
        }),

        // Keyword inclusion
        "word_count_keyword_3_times" => Some(InstructionVerifier {
            id: id.to_string(),
            check_fn: Arc::new(|text| keyword_count(text, "AI") >= 3),
        }),
        "word_count_keyword_4_times" => Some(InstructionVerifier {
            id: id.to_string(),
            check_fn: Arc::new(|text| keyword_count(text, "AI") >= 4),
        }),
        "word_count_keyword_2_times" => Some(InstructionVerifier {
            id: id.to_string(),
            check_fn: Arc::new(|text| keyword_count(text, "AI") >= 2),
        }),

        // Keyword exclusion
        "word_count_no_word_the" => Some(InstructionVerifier {
            id: id.to_string(),
            check_fn: Arc::new(|text| !keyword_count(text, "the").gt(&0)),
        }),
        "word_count_no_word_and" => Some(InstructionVerifier {
            id: id.to_string(),
            check_fn: Arc::new(|text| !keyword_count(text, "and").gt(&0)),
        }),

        // Structural constraints
        "list_3_items" => Some(InstructionVerifier {
            id: id.to_string(),
            check_fn: Arc::new(|text| contains_numbered_list(text, 3)),
        }),
        "list_2_items" => Some(InstructionVerifier {
            id: id.to_string(),
            check_fn: Arc::new(|text| contains_numbered_list(text, 2)),
        }),

        // Format constraints
        "write_in_all_caps" => Some(InstructionVerifier {
            id: id.to_string(),
            check_fn: Arc::new(|text| {
                let words: Vec<&str> = text.split_whitespace().collect();
                if words.is_empty() {
                    return false;
                }
                let upper = words
                    .iter()
                    .filter(|w| w.chars().any(|c| c.is_alphabetic()))
                    .count();
                upper as f64 / words.len() as f64 > 0.9
            }),
        }),
        "write_in_lowercase" => Some(InstructionVerifier {
            id: id.to_string(),
            check_fn: Arc::new(|text| {
                let words: Vec<&str> = text.split_whitespace().collect();
                if words.is_empty() {
                    return false;
                }
                let lower = words
                    .iter()
                    .filter(|w| w.chars().any(|c| c.is_alphabetic()))
                    .count();
                let alpha_words: Vec<&str> = words
                    .iter()
                    .filter(|w| w.chars().any(|c| c.is_alphabetic()))
                    .copied()
                    .collect();
                if alpha_words.is_empty() {
                    return false;
                }
                lower as f64 / alpha_words.len() as f64 > 0.9
            }),
        }),

        _ => None,
    }
}

fn word_count(text: &str) -> usize {
    text.split_whitespace().count()
}

fn keyword_count(text: &str, keyword: &str) -> usize {
    let text_lower = text.to_lowercase();
    let keyword_lower = keyword.to_lowercase();
    text_lower.matches(&keyword_lower).count()
}

fn contains_numbered_list(text: &str, count: usize) -> bool {
    let numbered = text
        .lines()
        .filter(|l| {
            let l = l.trim();
            l.starts_with(|c: char| c.is_numeric())
                || l.starts_with(|c: char| c.is_ascii_lowercase())
        })
        .count();
    numbered >= count
}

impl Benchmark for IFEvalBenchmark {
    fn name(&self) -> &str {
        "ifeval"
    }

    fn display_name(&self) -> &'static str {
        "IFEval"
    }

    fn category(&self) -> BenchmarkCategory {
        BenchmarkCategory::InstructionFollowing
    }

    fn pre_execute(&self, _config: &yaml_serde::Value) -> Result<()> {
        let dataset = load_ifeval_dataset()?;
        let mut state = self.state.lock().unwrap_or_else(|p| p.into_inner());
        state.items = dataset;
        state.current_idx = 0;
        Ok(())
    }

    fn execute_one(
        &self,
        model: &Model,
        _config: &yaml_serde::Value,
        tracker: &mut TokenTracker,
    ) -> Result<Option<TaskResult>> {
        let (row, idx) = {
            let mut state = self.state.lock().unwrap_or_else(|p| p.into_inner());
            if state.current_idx >= state.items.len() {
                return Ok(None);
            }
            let idx = state.current_idx;
            let row = state.items[idx].clone();
            state.current_idx += 1;
            (row, idx)
        };

        let verifiers = create_verifiers(&row.instruction_ids);

        // Skip if no verifiers implemented
        if verifiers.is_empty() {
            return Ok(Some(
                TaskResult::new(format!("task-{}", idx), false, 0.0, vec![]).with_metadata(Some(
                    serde_json::json!({
                        "skipped": true,
                        "instruction_ids": row.instruction_ids,
                        "reason": "no verifier implemented",
                    }),
                )),
            ));
        }

        let response = tracker.chat_completion(&model.model_name, "", &row.prompt)?;

        let mut followed = 0;
        let mut total = 0;
        let mut instruction_results = Vec::new();

        for verifier in verifiers {
            let passed = (verifier.check_fn)(&response);
            total += 1;
            if passed {
                followed += 1;
            }
            instruction_results.push(serde_json::json!({
                "instruction_id": verifier.id,
                "passed": passed,
            }));
        }

        let all_followed = followed == total;

        Ok(Some(
            TaskResult::new(
                format!("task-{}", idx),
                all_followed,
                followed as f64 / total as f64,
                row.instruction_ids.clone(),
            )
            .with_metadata(Some(serde_json::json!({
                "instruction_followed": followed,
                "instruction_total": total,
                "instruction_results": instruction_results,
            }))),
        ))
    }

    fn to_report_result(&self, b: &BenchmarkResult) -> Result<BenchmarkResult> {
        let raw = &b.raw;

        let (total_instructions, total_followed, skipped, output_tokens, thinking_tokens) = {
            if let Some(per_task) = raw.get("per_task").and_then(|v| v.as_array()) {
                let mut total = 0i64;
                let mut followed = 0i64;
                let mut skipped_count = 0i64;
                let out: i64 = per_task
                    .iter()
                    .filter_map(|t| t.get("output_tokens").and_then(|v| v.as_i64()))
                    .sum();
                let think: i64 = per_task
                    .iter()
                    .filter_map(|t| t.get("thinking_tokens").and_then(|v| v.as_i64()))
                    .sum();

                for task in per_task {
                    if task
                        .get("skipped")
                        .and_then(|v| v.as_bool())
                        .unwrap_or(false)
                    {
                        if let Some(ids) = task.get("instruction_ids").and_then(|v| v.as_array()) {
                            skipped_count += ids.len() as i64;
                        }
                    } else if let Some(meta) = task.get("metadata").and_then(|v| v.as_object()) {
                        let inst_total = meta
                            .get("instruction_total")
                            .and_then(|v| v.as_i64())
                            .unwrap_or(0);
                        let inst_followed = meta
                            .get("instruction_followed")
                            .and_then(|v| v.as_i64())
                            .unwrap_or(0);
                        total += inst_total;
                        followed += inst_followed;
                    }
                }
                (total, followed, skipped_count, out, think)
            } else {
                (
                    raw.get("total_instructions")
                        .and_then(|v| v.as_i64())
                        .unwrap_or(0),
                    raw.get("total_followed")
                        .and_then(|v| v.as_i64())
                        .unwrap_or(0),
                    raw.get("skipped_instructions")
                        .and_then(|v| v.as_i64())
                        .unwrap_or(0),
                    raw.get("output_tokens")
                        .and_then(|v| v.as_i64())
                        .unwrap_or(0),
                    raw.get("thinking_tokens")
                        .and_then(|v| v.as_i64())
                        .unwrap_or(0),
                )
            }
        };

        let follow_rate = if total_instructions == 0 {
            0.0
        } else {
            total_followed as f64 / total_instructions as f64
        };

        let mut scores = BTreeMap::new();
        scores.insert(
            "instruction_following_rate".to_string(),
            Score::float(follow_rate * 100.0, ScoreUnit::Percent)
                .primary(true)
                .higher_is_better(true),
        );
        scores.insert(
            "total_instructions".to_string(),
            Score::integer(total_instructions, ScoreUnit::Count),
        );
        scores.insert(
            "total_followed".to_string(),
            Score::integer(total_followed, ScoreUnit::Count),
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
                    "IFEval: {}/{} instructions followed ({:.1}%). {} instructions were skipped (no verifier implemented).",
                    total_followed, total_instructions, follow_rate * 100.0, skipped
                ),
            }],
            raw: raw.clone(),
        })
    }
}
