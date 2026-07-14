use crate::benchmarks::Benchmark;
use crate::client::Client;
use crate::config::Model;
use crate::download::download_with_retry;
use crate::reports::model::{BenchmarkResult, Score, ScoreUnit};
use anyhow::{Context, Result};
use serde::Deserialize;
use std::collections::BTreeMap;
use std::fs;
use std::sync::Arc;

pub struct IFEvalBenchmark;

const IFEVAL_URL: &str = "https://huggingface.co/datasets/google/IFEval/resolve/main/IFEval.json";

#[derive(Debug, Clone, Deserialize)]
struct IFEvalRow {
    prompt: String,
    #[serde(rename = "instruction_id")]
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
    let path = cache_dir.join("IFEval.json");

    if path.exists() {
        let content = fs::read_to_string(&path)?;
        return Ok(serde_json::from_str(&content)?);
    }

    fs::create_dir_all(&cache_dir)?;
    println!("  Downloading IFEval dataset...");
    let bytes = download_with_retry(IFEVAL_URL, 3)?
        .error_for_status()?
        .bytes()?;
    let tmp_path = path.with_extension(format!("json.tmp.{}", std::process::id()));
    fs::write(&tmp_path, bytes)?;
    fs::rename(&tmp_path, &path).context("failed to rename IFEval download")?;
    let content = fs::read_to_string(&path)?;
    Ok(serde_json::from_str(&content)?)
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

    fn category(&self) -> crate::reports::model::BenchmarkCategory {
        crate::reports::model::BenchmarkCategory::InstructionFollowing
    }

    fn pre_execute(&self, _config: &yaml_serde::Value) -> Result<()> {
        let _ = load_ifeval_dataset()?;
        Ok(())
    }

    fn execute(&self, model: &Model, _config: &yaml_serde::Value) -> Result<serde_json::Value> {
        let dataset = load_ifeval_dataset()?;
        let client = Client::new_with_model_params(&model.proxy, model.set_params.as_ref())?;

        let mut total_instructions = 0;
        let mut total_followed = 0;
        let mut instance_results = Vec::new();
        let mut total_output_tokens = 0u64;
        let mut total_thinking_tokens = 0u64;
        let mut skipped_instances = 0;

        for row in &dataset {
            let verifiers = create_verifiers(&row.instruction_ids);
            if verifiers.is_empty() {
                skipped_instances += row.instruction_ids.len();
                continue;
            }

            let prompt = row.prompt.clone();
            let (response, output_tokens, thinking_tokens) =
                client.chat_completion(&model.model_name, "", &prompt)?;

            total_output_tokens += output_tokens.unwrap_or(0);
            total_thinking_tokens += thinking_tokens.unwrap_or(0);

            let mut instance_followed = 0;
            let mut instruction_results = Vec::new();

            for verifier in verifiers {
                let passed = (verifier.check_fn)(&response);
                total_instructions += 1;
                if passed {
                    total_followed += 1;
                    instance_followed += 1;
                }
                instruction_results.push(serde_json::json!({
                    "instruction_id": verifier.id,
                    "passed": passed,
                }));
            }

            instance_results.push(serde_json::json!({
                "instance_id": row.instruction_ids.join("_"),
                "instructions_total": row.instruction_ids.len(),
                "instructions_followed": instance_followed,
                "instruction_results": instruction_results,
                "output_tokens": output_tokens,
                "thinking_tokens": thinking_tokens,
            }));
        }

        let follow_rate = if total_instructions == 0 {
            0.0
        } else {
            total_followed as f64 / total_instructions as f64
        };

        Ok(serde_json::json!({
            "instruction_following_rate": follow_rate,
            "total_instructions": total_instructions,
            "total_followed": total_followed,
            "skipped_instructions": skipped_instances,
            "output_tokens": total_output_tokens,
            "thinking_tokens": total_thinking_tokens,
            "instance_results": instance_results,
        }))
    }

    fn to_report_result(&self, raw: &serde_json::Value) -> Result<BenchmarkResult> {
        let follow_rate = raw
            .get("instruction_following_rate")
            .and_then(|v| v.as_f64())
            .unwrap_or(0.0);
        let total_instructions = raw
            .get("total_instructions")
            .and_then(|v| v.as_i64())
            .unwrap_or(0);
        let total_followed = raw
            .get("total_followed")
            .and_then(|v| v.as_i64())
            .unwrap_or(0);
        let skipped = raw
            .get("skipped_instructions")
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

        let mut scores = BTreeMap::new();
        scores.insert(
            "instruction_following_rate".to_string(),
            Score::float(follow_rate, ScoreUnit::Percent)
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
