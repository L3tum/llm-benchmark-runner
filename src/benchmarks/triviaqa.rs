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

pub struct TriviaQABenchmark;

static DATASET: OnceLock<Vec<TriviaQARow>> = OnceLock::new();

#[derive(Debug, Clone, Deserialize)]
struct TriviaQARow {
    question: String,
    entity_pages: Option<Vec<TriviaQAEntity>>,
    question_source: Option<String>,
}

#[derive(Debug, Clone, Deserialize)]
struct TriviaQAEntity {
    title: String,
}

fn load_trivia_qa() -> &'static Vec<TriviaQARow> {
    DATASET.get_or_init(|| {
        let cache_dir = dirs::cache_dir()
            .unwrap_or_default()
            .join("llm-benchmark-runner")
            .join("trivia_qa");
        let path = cache_dir.join("rc.nocontext.json");

        if path.exists() {
            let content = fs::read_to_string(&path).expect("Failed to read cached TriviaQA");
            return serde_json::from_str(&content).expect("Failed to parse TriviaQA");
        }

        fs::create_dir_all(&cache_dir).expect("Failed to create cache dir");
        println!("  Downloading TriviaQA (rc.nocontext) dataset...");
        let url =
            "https://huggingface.co/datasets/mandarjoshi/trivia_qa/resolve/main/rc/nocontext.json";
        let bytes = download_with_retry_bytes(url, 3, 60, "llm-benchmark-runner")
            .expect("Failed to download TriviaQA");

        let dataset: TriviaQADataset = serde_json::from_slice(&bytes).unwrap();
        let rows = dataset.data;
        fs::write(&path, &bytes).expect("Failed to save TriviaQA");
        rows
    })
}

#[derive(Debug, Deserialize)]
struct TriviaQADataset {
    data: Vec<TriviaQARow>,
}

impl Benchmark for TriviaQABenchmark {
    fn name(&self) -> &str {
        "triviaqa"
    }

    fn display_name(&self) -> &'static str {
        "TriviaQA (Closed-Book)"
    }

    fn category(&self) -> BenchmarkCategory {
        BenchmarkCategory::Knowledge
    }

    fn pre_execute(&self, _config: &yaml_serde::Value) -> Result<()> {
        let _ = load_trivia_qa();
        Ok(())
    }

    fn execute(&self, model: &Model, _config: &yaml_serde::Value) -> Result<BenchmarkResult> {
        let dataset = load_trivia_qa();
        let client = Client::new_with_model_params(&model.proxy, model.set_params.as_ref())?;

        let system_prompt =
            "You are a trivia expert. Answer with a single factual entity (person, place, thing).";

        let user_prompt = r#"Question: Who discovered penicillin?
Answer: Alexander Fleming

Question: What is the largest ocean on Earth?
Answer: Pacific Ocean

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

            let response = response.trim().to_lowercase();
            // Check if response matches any entity page title (case-insensitive)
            let is_correct = item
                .entity_pages
                .as_ref()
                .map(|entities| {
                    entities.iter().any(|e| {
                        let title = e.title.to_lowercase();
                        response.contains(&title) || title.contains(&response)
                    })
                })
                .unwrap_or(false);

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
                    "TriviaQA: {} correct out of {} ({:.1}%)",
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
                    "TriviaQA: {} correct out of {} ({:.1}%)",
                    correct,
                    total,
                    em_score * 100.0
                ),
            }],
            raw: raw.clone(),
        })
    }
}
