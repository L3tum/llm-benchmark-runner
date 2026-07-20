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

pub struct FeverBenchmark;

static DATASET: OnceLock<Vec<FeverItem>> = OnceLock::new();

#[derive(Debug, Clone, Deserialize)]
struct FeverItem {
    claim: String,
    label: String, // "SUPPORTS", "REFUTES", "NOT ENOUGH INFO"
}

fn load_fever_dataset() -> &'static Vec<FeverItem> {
    DATASET.get_or_init(|| {
        let cache_dir = dirs::cache_dir()
            .unwrap_or_default()
            .join("llm-benchmark-runner")
            .join("fever");
        let path = cache_dir.join("fever_dev.json");

        if path.exists() {
            let content = fs::read_to_string(&path).expect("Failed to read cached FEVER");
            return serde_json::from_str(&content).expect("Failed to parse FEVER");
        }

        fs::create_dir_all(&cache_dir).expect("Failed to create cache dir");
        println!("  Downloading FEVER dev dataset...");

        // Try multiple sources
        let urls = [
            // Official website (often blocks automated access)
            "https://fever.ai/data/fever_dev.json",
            // HuggingFace (requires datasets library, may not have direct file)
            "https://huggingface.co/datasets/fever/fever/resolve/main/data/paper_dev.json",
            // GitHub mirror (if available)
            "https://raw.githubusercontent.com/awslabs/fever/main/data/fever_dev.json",
        ];

        let mut last_err = None;
        for url in &urls {
            match download_with_retry_bytes(url, 2, 30, "llm-benchmark-runner") {
                Ok(bytes) => {
                    // Try to parse
                    match serde_json::from_slice::<Vec<FeverItem>>(&bytes) {
                        Ok(items) => {
                            fs::write(&path, bytes).expect("Failed to save FEVER");
                            return items;
                        }
                        Err(_) => {
                            // Maybe it's a wrapper object with "dev" key (from HuggingFace)
                            if let Ok(parsed) = serde_json::from_slice::<serde_json::Value>(&bytes)
                            {
                                if let Some(dev) = parsed.get("dev") {
                                    if let Some(claims) =
                                        dev.get("claims").and_then(|c| c.as_array())
                                    {
                                        let mut items = Vec::new();
                                        for claim_obj in claims {
                                            items.push(FeverItem {
                                                claim: claim_obj
                                                    .get("claim")
                                                    .and_then(|c| c.as_str())
                                                    .unwrap_or("")
                                                    .to_string(),
                                                label: claim_obj
                                                    .get("label")
                                                    .and_then(|l| l.as_str())
                                                    .unwrap_or("NOT ENOUGH INFO")
                                                    .to_string(),
                                            });
                                        }
                                        fs::write(&path, bytes).expect("Failed to save FEVER");
                                        return items;
                                    }
                                }
                            }
                            eprintln!("  Failed to parse FEVER from {}", url);
                        }
                    }
                }
                Err(e) => {
                    last_err = Some(anyhow::anyhow!("Failed to download from {}: {}", url, e));
                }
            }
        }

        // If all downloads fail, provide a helpful error message
        let err = last_err.unwrap_or(anyhow::anyhow!("No download sources available"));
        eprintln!("  Error: {}", err);
        eprintln!(
            "  Please manually download the FEVER dataset from https://fever.ai and place it at:"
        );
        eprintln!("  {}", path.display());
        std::process::exit(1);
    })
}

impl Benchmark for FeverBenchmark {
    fn name(&self) -> &str {
        "fever"
    }

    fn display_name(&self) -> &'static str {
        "FEVER (Fact Extraction and VERification)"
    }

    fn category(&self) -> BenchmarkCategory {
        BenchmarkCategory::Hallucination
    }

    fn pre_execute(&self, _config: &yaml_serde::Value) -> Result<()> {
        let _ = load_fever_dataset();
        Ok(())
    }

    fn execute(&self, model: &Model, _config: &yaml_serde::Value) -> Result<BenchmarkResult> {
        let dataset = load_fever_dataset();
        let client = Client::new_with_model_params(&model.proxy, model.set_params.as_ref())?;

        let system_prompt = "You are a fact verification assistant. Given a claim, determine whether it is SUPPORTS (the claim is true based on factual knowledge), REFUTES (the claim is false), or NOT ENOUGH INFO (you cannot determine its truth from your knowledge). Respond with only one of these three labels.";

        // 16-shot prompt examples (simplified to 5 for brevity)
        let user_prompt = r#"Claim: The Eiffel Tower is located in Paris.
Label: SUPPORTS

Claim: Water boils at 100 degrees Celsius at sea level.
Label: SUPPORTS

Claim: Albert Einstein was born in 1879.
Label: SUPPORTS

Claim: The moon is made of cheese.
Label: REFUTES

Claim: The Earth is flat.
Label: REFUTES

Claim: A certain unverified conspiracy theory about a famous person.
Label: NOT ENOUGH INFO

Claim: {claim}
Label:"#;

        let total = dataset.len();
        let mut correct = 0;
        let mut output_tokens_total: i64 = 0;
        let mut thinking_tokens_total: i64 = 0;
        let mut label_stats: BTreeMap<String, Vec<bool>> = BTreeMap::new();

        for item in dataset {
            let prompt = user_prompt.replace("{claim}", &item.claim);
            let (response, output_tokens, thinking_tokens) =
                client.chat_completion(&model.model_name, system_prompt, &prompt)?;

            output_tokens_total += output_tokens.unwrap_or(0) as i64;
            thinking_tokens_total += thinking_tokens.unwrap_or(0) as i64;

            let response = response.trim().to_uppercase();
            let is_correct = response.contains("SUPPORTS") && item.label == "SUPPORTS"
                || response.contains("REFUTES") && item.label == "REFUTES"
                || response.contains("NOT ENOUGH INFO") && item.label == "NOT ENOUGH INFO";

            if is_correct {
                correct += 1;
            }

            label_stats
                .entry(item.label.clone())
                .or_default()
                .push(is_correct);
        }

        let accuracy = correct as f64 / total as f64;

        // Per-label breakdown
        let mut breakdowns = BTreeMap::new();
        for (label, results) in label_stats {
            let label_correct: i64 = results.iter().filter(|&&c| c).count() as i64;
            let label_total = results.len() as i64;
            let label_str = label.clone();
            breakdowns.insert(
                label_str.clone(),
                crate::reports::model::BreakdownTable {
                    title: label_str,
                    rows: BTreeMap::from_iter([
                        (
                            "accuracy".to_string(),
                            BTreeMap::from_iter([(
                                "accuracy".to_string(),
                                Score::float(
                                    label_correct as f64 / label_total as f64 * 100.0,
                                    ScoreUnit::Percent,
                                ),
                            )]),
                        ),
                        (
                            "count".to_string(),
                            BTreeMap::from_iter([(
                                "total".to_string(),
                                Score::integer(label_total, ScoreUnit::Count),
                            )]),
                        ),
                    ]),
                },
            );
        }

        let raw_json = serde_json::json!({
            "accuracy": accuracy,
            "total": total,
            "correct": correct,
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
                    "FEVER: {}/{} correct ({:.1}%)",
                    correct,
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
                    "FEVER: {}/{} correct ({:.1}%)",
                    correct,
                    total,
                    accuracy * 100.0
                ),
            }],
            raw: raw.clone(),
        })
    }
}
