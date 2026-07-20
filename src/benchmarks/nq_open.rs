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

pub struct NQOpenBenchmark;

static DATASET: OnceLock<Vec<NQOpenItem>> = OnceLock::new();

#[derive(Debug, Clone, Deserialize)]
struct NQOpenItem {
    question: String,
    answers: Vec<String>,
}

fn load_nq_open() -> &'static Vec<NQOpenItem> {
    DATASET.get_or_init(|| {
        let cache_dir = dirs::cache_dir()
            .unwrap_or_default()
            .join("llm-benchmark-runner")
            .join("nq_open");
        let path = cache_dir.join("NQ-Open.json");

        if path.exists() {
            let content = fs::read_to_string(&path).expect("Failed to read cached NQ-Open");
            return serde_json::from_str(&content).expect("Failed to parse NQ-Open");
        }

        fs::create_dir_all(&cache_dir).expect("Failed to create cache dir");
        println!("  Downloading NQ-Open dataset...");
        let url = "https://huggingface.co/datasets/nq_open/resolve/main/data/test.csv";
        let bytes = download_with_retry_bytes(url, 3, 60, "llm-benchmark-runner")
            .expect("Failed to download NQ-Open");

        // Parse CSV using csv crate for proper quoted field handling
        let content = String::from_utf8(Vec::from(bytes.as_ref())).expect("Failed to decode UTF-8");
        let mut reader = csv::ReaderBuilder::new()
            .delimiter(b',')
            .has_headers(true)
            .from_reader(content.as_bytes());
        let mut items = Vec::new();
        for record in reader.records().flatten() {
            if record.len() >= 2 {
                let question = record.get(0).unwrap_or("").to_string();
                let answers_str = record.get(1).unwrap_or("").to_string();
                let answers = answers_str
                    .split("\\t")
                    .map(|s| s.trim().to_string())
                    .filter(|s| !s.is_empty())
                    .collect();
                items.push(NQOpenItem { question, answers });
            }
        }

        fs::write(&path, &bytes).expect("Failed to save NQ-Open");
        items
    })
}

impl Benchmark for NQOpenBenchmark {
    fn name(&self) -> &str {
        "nq_open"
    }

    fn display_name(&self) -> &'static str {
        "NQ Open (Natural Questions)"
    }

    fn category(&self) -> BenchmarkCategory {
        BenchmarkCategory::Knowledge
    }

    fn pre_execute(&self, _config: &yaml_serde::Value) -> Result<()> {
        let _ = load_nq_open();
        Ok(())
    }

    fn execute(&self, model: &Model, _config: &yaml_serde::Value) -> Result<BenchmarkResult> {
        let dataset = load_nq_open();
        let client = Client::new_with_model_params(&model.proxy, model.set_params.as_ref())?;

        let system_prompt = "You are a knowledgeable assistant. Answer the question with a single entity (person, place, organization, etc.) or a short factual phrase.";

        let user_prompt = r#"Question: Who is the CEO of Tesla?
Answer: Elon Musk

Question: What is the capital of France?
Answer: Paris

Question: {question}
Answer:"#;

        let total = dataset.len();
        let mut exact_match = 0;
        let mut output_tokens_total: i64 = 0;
        let mut thinking_tokens_total: i64 = 0;

        for item in dataset {
            let prompt = user_prompt.replace("{question}", &item.question);
            let (response, output_tokens, thinking_tokens) =
                client.chat_completion(&model.model_name, system_prompt, &prompt)?;

            output_tokens_total += output_tokens.unwrap_or(0) as i64;
            thinking_tokens_total += thinking_tokens.unwrap_or(0) as i64;

            let response = response.trim();
            // Exact match: response matches any of the accepted answers (case-insensitive)
            let is_correct = item.answers.iter().any(|a| {
                let a = a.trim().to_lowercase();
                response.to_lowercase() == a
            });

            if is_correct {
                exact_match += 1;
            }
        }

        let em_score = exact_match as f64 / total as f64;
        let raw_json = serde_json::json!({
            "exact_match": em_score,
            "total": total,
            "correct": exact_match,
            "output_tokens": output_tokens_total,
            "thinking_tokens": thinking_tokens_total,
        });

        Ok(BenchmarkResult {
            scores: BTreeMap::new(),
            breakdowns: BTreeMap::new(),
            error_classification: BTreeMap::new(),
            artifacts: vec![],
            diagnostics: vec![crate::reports::model::Diagnostic {
                level: "info".to_string(),
                message: format!(
                    "NQ Open: {} exact matches out of {} ({:.1}%)",
                    exact_match,
                    total,
                    em_score * 100.0
                ),
            }],
            raw: raw_json,
        })
    }

    fn to_report_result(&self, b: &BenchmarkResult) -> Result<BenchmarkResult> {
        let raw = &b.raw;
        let em_score = raw
            .get("exact_match")
            .and_then(|v| v.as_f64())
            .unwrap_or(0.0);
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
            "exact_match".to_string(),
            Score::float(em_score * 100.0, ScoreUnit::Percent)
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
            breakdowns: BTreeMap::new(),
            error_classification: BTreeMap::new(),
            artifacts: vec![],
            diagnostics: vec![crate::reports::model::Diagnostic {
                level: "info".to_string(),
                message: format!(
                    "NQ Open: {} exact matches out of {} ({:.1}%)",
                    correct,
                    total,
                    em_score * 100.0
                ),
            }],
            raw: raw.clone(),
        })
    }
}
