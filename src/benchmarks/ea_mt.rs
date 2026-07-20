use crate::benchmarks::Benchmark;
use crate::client::Client;
use crate::config::Model;
use crate::download::download_with_retry_bytes;
use crate::reports::model::{BenchmarkCategory, BenchmarkResult, Score, ScoreUnit};
use anyhow::Result;
use serde::Deserialize;
use std::collections::BTreeMap;
use std::fs;
use std::sync::OnceLock;

pub struct EAMTBenchmark;

static DATASET: OnceLock<Vec<EAItem>> = OnceLock::new();

#[derive(Debug, Clone, Deserialize)]
struct EAItem {
    sentence_id: String,
    source_language: String,
    target_language: String,
    sentence: String, // source sentence
    target: String,   // reference translation
    entities: Vec<Entity>,
}

#[derive(Debug, Clone, Deserialize)]
struct Entity {
    entity: String,
    #[serde(rename = "entity_type")]
    entity_type: String,
    translation: Vec<String>,
}

fn load_eamt_dataset() -> &'static Vec<EAItem> {
    DATASET.get_or_init(|| {
        let cache_dir = dirs::cache_dir()
            .unwrap_or_default()
            .join("llm-benchmark-runner")
            .join("ea_mt");
        let path = cache_dir.join("ea-mt-benchmark.json");

        if path.exists() {
            let content = fs::read_to_string(&path).expect("Failed to read cached EA-MT");
            let parsed: EAMTDataset =
                serde_json::from_str(&content).expect("Failed to parse EA-MT");
            return parsed.data;
        }

        fs::create_dir_all(&cache_dir).expect("Failed to create cache dir");
        println!("  Downloading EA-MT (Entity-Aware Machine Translation) dataset...");
        let url = "https://huggingface.co/datasets/sapienzanlp/ea-mt-benchmark/resolve/main/ea-mt-benchmark.json";
        let bytes = download_with_retry_bytes(url, 3, 60, "llm-benchmark-runner")
            .expect("Failed to download EA-MT");

        let parsed: EAMTDataset = serde_json::from_slice(&bytes).expect("Failed to parse EA-MT");
        fs::write(&path, &bytes).expect("Failed to save EA-MT");
        parsed.data
    })
}

#[derive(Debug, Deserialize)]
struct EAMTDataset {
    data: Vec<EAItem>,
}

impl Benchmark for EAMTBenchmark {
    fn name(&self) -> &str {
        "ea_mt"
    }

    fn display_name(&self) -> &'static str {
        "EA-MT (Entity-Aware Machine Translation)"
    }

    fn category(&self) -> BenchmarkCategory {
        BenchmarkCategory::Translation
    }

    fn pre_execute(&self, _config: &yaml_serde::Value) -> Result<()> {
        let _ = load_eamt_dataset();
        Ok(())
    }

    fn execute(&self, model: &Model, _config: &yaml_serde::Value) -> Result<BenchmarkResult> {
        let dataset = load_eamt_dataset();
        let client = Client::new_with_model_params(&model.proxy, model.set_params.as_ref())?;

        let system_prompt = "You are a translation expert. Translate the given sentence into the target language. Pay special attention to named entities — translate them correctly if appropriate.";

        let user_prompt = r#"Translate from {source_language} to {target_language}:

Source: {sentence}
Translation:"#;

        let total = dataset.len();
        let mut exact_match = 0;
        let mut output_tokens_total: i64 = 0;
        let mut thinking_tokens_total: i64 = 0;
        let mut lang_pair_stats: BTreeMap<String, Vec<bool>> = BTreeMap::new();

        for item in dataset {
            let lang_pair = format!("{}-{}", item.source_language, item.target_language);
            let prompt = user_prompt
                .replace("{source_language}", &item.source_language)
                .replace("{target_language}", &item.target_language)
                .replace("{sentence}", &item.sentence);

            let (response, output_tokens, thinking_tokens) =
                client.chat_completion(&model.model_name, system_prompt, &prompt)?;

            output_tokens_total += output_tokens.unwrap_or(0) as i64;
            thinking_tokens_total += thinking_tokens.unwrap_or(0) as i64;

            let response = response.trim().to_lowercase();
            let reference = item.target.trim().to_lowercase();

            let full_match = response == reference;
            if full_match {
                exact_match += 1;
            }

            lang_pair_stats
                .entry(lang_pair)
                .or_default()
                .push(full_match);
        }

        let accuracy = exact_match as f64 / total as f64;

        // Language pair breakdown
        let mut breakdowns = BTreeMap::new();
        for (pair, results) in lang_pair_stats {
            let correct_count: i64 = results.iter().filter(|&&c| c).count() as i64;
            let total_count = results.len() as i64;
            let pair_str = pair.clone();
            breakdowns.insert(
                pair_str.clone(),
                crate::reports::model::BreakdownTable {
                    title: pair_str,
                    rows: BTreeMap::from_iter([
                        (
                            "accuracy".to_string(),
                            BTreeMap::from_iter([(
                                "accuracy".to_string(),
                                Score::float(
                                    correct_count as f64 / total_count as f64 * 100.0,
                                    ScoreUnit::Percent,
                                ),
                            )]),
                        ),
                        (
                            "count".to_string(),
                            BTreeMap::from_iter([(
                                "total".to_string(),
                                Score::integer(total_count, ScoreUnit::Count),
                            )]),
                        ),
                    ]),
                },
            );
        }

        let raw_json = serde_json::json!({
            "accuracy": accuracy,
            "total": total,
            "correct": exact_match,
            "output_tokens": output_tokens_total,
            "thinking_tokens": thinking_tokens_total,
        });

        Ok(BenchmarkResult {
            scores: BTreeMap::new(),
            breakdowns,
            error_classification: BTreeMap::new(),
            artifacts: vec![],
            diagnostics: vec![crate::reports::model::Diagnostic {
                level: "info".to_string(),
                message: format!(
                    "EA-MT: {}/{} correct ({:.1}%)",
                    exact_match,
                    total,
                    accuracy * 100.0
                ),
            }],
            raw: raw_json,
        })
    }

    fn to_report_result(&self, b: &BenchmarkResult) -> Result<BenchmarkResult> {
        let raw = &b.raw;
        let accuracy = raw.get("accuracy").and_then(|v| v.as_f64()).unwrap_or(0.0);
        let total = raw.get("total").and_then(|v| v.as_i64()).unwrap_or(0);
        let correct = raw.get("correct").and_then(|v| v.as_i64()).unwrap_or(0);
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

        Ok(BenchmarkResult {
            scores,
            breakdowns: b.breakdowns.clone(),
            error_classification: BTreeMap::new(),
            artifacts: vec![],
            diagnostics: vec![crate::reports::model::Diagnostic {
                level: "info".to_string(),
                message: format!(
                    "EA-MT: {}/{} correct ({:.1}%)",
                    correct,
                    total,
                    accuracy * 100.0
                ),
            }],
            raw: raw.clone(),
        })
    }
}
