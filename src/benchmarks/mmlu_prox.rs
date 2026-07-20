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

pub struct MmluProxBenchmark;

static DATASET: OnceLock<Vec<MmluProXItem>> = OnceLock::new();

#[derive(Debug, Clone, Deserialize)]
struct MmluProXItem {
    id: String,
    language: String,
    category: String,
    question: String,
    choices: Vec<String>,
    correct_answer: String, // single letter
    subject: String,
}

fn load_mmlu_prox() -> &'static Vec<MmluProXItem> {
    DATASET.get_or_init(|| {
        let cache_dir = dirs::cache_dir()
            .unwrap_or_default()
            .join("llm-benchmark-runner")
            .join("mmlu_prox");
        let path = cache_dir.join("MMLU-ProX.json");

        if path.exists() {
            let content = fs::read_to_string(&path).expect("Failed to read cached MMLU-ProX");
            return serde_json::from_str(&content).expect("Failed to parse MMLU-ProX");
        }

        fs::create_dir_all(&cache_dir).expect("Failed to create cache dir");
        println!("  Downloading MMLU-ProX multilingual dataset...");
        let url = "https://huggingface.co/datasets/li-lab/MMLU-ProX/resolve/main/MMLU-ProX.json";
        let bytes = download_with_retry_bytes(url, 3, 60, "llm-benchmark-runner")
            .expect("Failed to download MMLU-ProX");

        let dataset: MmluProXDataset = serde_json::from_slice(&bytes).unwrap();
        let items = dataset.data;
        fs::write(&path, &bytes).expect("Failed to save MMLU-ProX");
        items
    })
}

#[derive(Debug, Deserialize)]
struct MmluProXDataset {
    data: Vec<MmluProXItem>,
}

impl Benchmark for MmluProxBenchmark {
    fn name(&self) -> &str {
        "mmlu_prox"
    }

    fn display_name(&self) -> &'static str {
        "MMLU-ProX (Multilingual)"
    }

    fn category(&self) -> BenchmarkCategory {
        BenchmarkCategory::Knowledge
    }

    fn pre_execute(&self, _config: &yaml_serde::Value) -> Result<()> {
        let _ = load_mmlu_prox();
        Ok(())
    }

    fn execute(&self, model: &Model, _config: &yaml_serde::Value) -> Result<BenchmarkResult> {
        let dataset = load_mmlu_prox();
        let client = Client::new_with_model_params(&model.proxy, model.set_params.as_ref())?;

        let system_prompt = "You are a multilingual multiple-choice question expert. Select the single correct answer from the options. Respond with only the letter.";

        let user_prompt = r#"Question: What is the capital of France?
A. Paris
B. London
C. Berlin
Answer: A

Question: What is the largest planet?
A. Mars
B. Jupiter
C. Earth
Answer: B

Question: {question}
{choices}
Answer:"#;

        let total = dataset.len();
        let mut correct = 0;
        let mut output_tokens_total: i64 = 0;
        let mut thinking_tokens_total: i64 = 0;
        let mut lang_stats: BTreeMap<String, Vec<bool>> = BTreeMap::new();

        for item in dataset {
            let mut choices_str = String::new();
            let labels: Vec<char> = ('A'..='J').collect();
            for (i, choice) in item.choices.iter().enumerate() {
                choices_str.push_str(&format!("{}. {}\n", labels[i], choice));
            }

            let prompt = user_prompt
                .replace("{question}", &item.question)
                .replace("{choices}", &choices_str);

            let (response, output_tokens, thinking_tokens) =
                client.chat_completion(&model.model_name, system_prompt, &prompt)?;

            output_tokens_total += output_tokens.unwrap_or(0) as i64;
            thinking_tokens_total += thinking_tokens.unwrap_or(0) as i64;

            let response_letter = response.chars().next().unwrap_or('Z').to_ascii_uppercase();
            let expected = item
                .correct_answer
                .chars()
                .next()
                .unwrap_or('A')
                .to_ascii_uppercase();
            let is_correct = response_letter == expected;

            if is_correct {
                correct += 1;
            }

            lang_stats
                .entry(item.language.clone())
                .or_default()
                .push(is_correct);
        }

        let accuracy = correct as f64 / total as f64;

        // Language breakdown
        let mut breakdowns = BTreeMap::new();
        for (lang, results) in lang_stats {
            let lang_correct: i64 = results.iter().filter(|&&c| c).count() as i64;
            let lang_total = results.len() as i64;
            let lang_str = lang.clone();
            breakdowns.insert(
                lang_str.clone(),
                crate::reports::model::BreakdownTable {
                    title: lang_str,
                    rows: BTreeMap::from_iter([
                        (
                            "accuracy".to_string(),
                            BTreeMap::from_iter([(
                                "accuracy".to_string(),
                                Score::float(
                                    lang_correct as f64 / lang_total as f64 * 100.0,
                                    ScoreUnit::Percent,
                                ),
                            )]),
                        ),
                        (
                            "count".to_string(),
                            BTreeMap::from_iter([(
                                "total".to_string(),
                                Score::integer(lang_total, ScoreUnit::Count),
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
                    "MMLU-ProX: {}/{} correct ({:.1}%)",
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
                    "MMLU-ProX: {}/{} correct ({:.1}%)",
                    correct,
                    total,
                    accuracy * 100.0
                ),
            }],
            raw: raw.clone(),
        })
    }
}
