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

pub struct SquadV2Benchmark;

static DATASET: OnceLock<Vec<SquadV2Item>> = OnceLock::new();

#[derive(Debug, Clone, Deserialize)]
struct SquadV2Item {
    title: String,
    context: String,
    question: String,
    answers: Option<Vec<SquadV2Answer>>,
    is_impossible: Option<bool>,
}

#[derive(Debug, Clone, Deserialize)]
struct SquadV2Answer {
    text: String,
    answer_start: i64,
}

fn load_squad_v2() -> &'static Vec<SquadV2Item> {
    DATASET.get_or_init(|| {
        let cache_dir = dirs::cache_dir()
            .unwrap_or_default()
            .join("llm-benchmark-runner")
            .join("squad_v2");
        let path = cache_dir.join("SQuAD2.0.json");

        if path.exists() {
            let content = fs::read_to_string(&path).expect("Failed to read cached SQuAD 2.0");
            let parsed: SQuADDataset =
                serde_json::from_str(&content).expect("Failed to parse SQuAD 2.0");
            let mut items = Vec::new();
            for data_item in &parsed.data {
                for paragraph in &data_item.paragraphs {
                    for qa in &paragraph.qas {
                        items.push(SquadV2Item {
                            title: data_item.title.clone(),
                            context: paragraph.context.clone(),
                            question: qa.question.clone(),
                            answers: qa.answers.clone(),
                            is_impossible: qa.is_impossible,
                        });
                    }
                }
            }
            return items;
        }

        fs::create_dir_all(&cache_dir).expect("Failed to create cache dir");
        println!("  Downloading SQuAD 2.0 dataset...");
        let url =
            "https://huggingface.co/datasets/rajpurkar/squad_v2/resolve/main/data/dev-v2.0.json";
        let bytes = download_with_retry_bytes(url, 3, 60, "llm-benchmark-runner")
            .expect("Failed to download SQuAD 2.0");

        let parsed: SQuADDataset =
            serde_json::from_slice(&bytes).expect("Failed to parse SQuAD 2.0");
        let mut items = Vec::new();
        for data_item in &parsed.data {
            for paragraph in &data_item.paragraphs {
                for qa in &paragraph.qas {
                    items.push(SquadV2Item {
                        title: data_item.title.clone(),
                        context: paragraph.context.clone(),
                        question: qa.question.clone(),
                        answers: qa.answers.clone(),
                        is_impossible: qa.is_impossible,
                    });
                }
            }
        }

        fs::write(&path, &bytes).expect("Failed to save SQuAD 2.0");
        items
    })
}

#[derive(Debug, Deserialize)]
struct SQuADDataset {
    data: Vec<SQuADDataItem>,
}

#[derive(Debug, Deserialize)]
struct SQuADDataItem {
    title: String,
    paragraphs: Vec<SQuADParagraph>,
}

#[derive(Debug, Deserialize)]
struct SQuADParagraph {
    context: String,
    qas: Vec<SQuADQA>,
}

#[derive(Debug, Deserialize)]
struct SQuADQA {
    question: String,
    #[serde(default)]
    is_impossible: Option<bool>,
    answers: Option<Vec<SquadV2Answer>>,
}

impl Benchmark for SquadV2Benchmark {
    fn name(&self) -> &str {
        "squad_v2"
    }

    fn display_name(&self) -> &'static str {
        "SQuAD 2.0"
    }

    fn category(&self) -> BenchmarkCategory {
        BenchmarkCategory::Hallucination
    }

    fn pre_execute(&self, _config: &yaml_serde::Value) -> Result<()> {
        let _ = load_squad_v2();
        Ok(())
    }

    fn execute(&self, model: &Model, _config: &yaml_serde::Value) -> Result<BenchmarkResult> {
        let dataset = load_squad_v2();
        let client = Client::new_with_model_params(&model.proxy, model.set_params.as_ref())?;

        let system_prompt = "You are a reading comprehension assistant. Answer the question using ONLY the provided context. If the answer cannot be found in the context, respond with 'no answer'.";

        let user_prompt = r#"Context: {context}
Question: {question}
Answer:"#;

        let total = dataset.len();
        let mut exact_match_total = 0;
        let mut f1_total: f64 = 0.0;
        let mut output_tokens_total: i64 = 0;
        let mut thinking_tokens_total: i64 = 0;
        let mut unanswerable_correct = 0;
        let mut unanswerable_total = 0;

        for item in dataset {
            let prompt = user_prompt
                .replace("{context}", &item.context)
                .replace("{question}", &item.question);

            let (response, output_tokens, thinking_tokens) =
                client.chat_completion(&model.model_name, system_prompt, &prompt)?;

            output_tokens_total += output_tokens.unwrap_or(0) as i64;
            thinking_tokens_total += thinking_tokens.unwrap_or(0) as i64;

            let response = response.trim();
            let is_unanswerable = item.is_impossible.unwrap_or(false);

            if is_unanswerable {
                unanswerable_total += 1;
                // For unanswerable questions, the correct response is "no answer"
                let is_correct = response.to_lowercase().contains("no answer")
                    || response.to_lowercase().contains("not answerable")
                    || response.to_lowercase().contains("cannot be found");
                if is_correct {
                    unanswerable_correct += 1;
                }
            } else if let Some(ref answers) = item.answers {
                if !answers.is_empty() {
                    let response_lower = response.trim().to_lowercase();
                    let best_em = answers
                        .iter()
                        .any(|a| a.text.trim().to_lowercase() == response_lower);

                    let best_f1 = answers
                        .iter()
                        .map(|a| compute_f1(&a.text, response))
                        .fold(0.0f64, f64::max);
                    f1_total += best_f1 * 100.0;
                    if best_em {
                        exact_match_total += 1;
                    }
                }
            }
        }

        let em_score = exact_match_total as f64 / total as f64;
        let f1_score = f1_total / total as f64;
        let unanswerable_acc = if unanswerable_total > 0 {
            unanswerable_correct as f64 / unanswerable_total as f64
        } else {
            0.0
        };

        let raw_json = serde_json::json!({
            "exact_match": em_score,
            "f1": f1_score,
            "total": total,
            "exact_match_correct": exact_match_total,
            "unanswerable_total": unanswerable_total,
            "unanswerable_correct": unanswerable_correct,
            "output_tokens": output_tokens_total,
            "thinking_tokens": thinking_tokens_total,
        });

        Ok(BenchmarkResult {
            scores: BTreeMap::new(),
            breakdowns: BTreeMap::new(),
            error_classification: BTreeMap::new(),
            artifacts: vec![],
            diagnostics: vec![
                crate::reports::model::Diagnostic {
                    level: "info".to_string(),
                    message: format!(
                        "SQuAD 2.0: EM {:.1}%, F1 {:.1}% (answerable), unanswerable accuracy {:.1}% ({}/{})",
                        em_score, f1_score, unanswerable_acc * 100.0, unanswerable_correct, unanswerable_total
                    ),
                }
            ],
            raw: raw_json,
        })
    }

    fn to_report_result(&self, b: &BenchmarkResult) -> Result<BenchmarkResult> {
        let raw = &b.raw;
        let em_score = raw
            .get("exact_match")
            .and_then(|v| v.as_f64())
            .unwrap_or(0.0);
        let f1_score = raw.get("f1").and_then(|v| v.as_f64()).unwrap_or(0.0);
        let total = raw.get("total").and_then(|v| v.as_i64()).unwrap_or(0);
        let _exact_match_correct = raw
            .get("exact_match_correct")
            .and_then(|v| v.as_i64())
            .unwrap_or(0);
        let unanswerable_total = raw
            .get("unanswerable_total")
            .and_then(|v| v.as_i64())
            .unwrap_or(0);
        let unanswerable_correct = raw
            .get("unanswerable_correct")
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
            "exact_match".to_string(),
            Score::float(em_score * 100.0, ScoreUnit::Percent)
                .primary(true)
                .higher_is_better(true),
        );
        scores.insert(
            "f1".to_string(),
            Score::float(f1_score * 100.0, ScoreUnit::Percent),
        );
        scores.insert(
            "unanswerable_accuracy".to_string(),
            Score::float(
                if unanswerable_total > 0 {
                    unanswerable_correct as f64 / unanswerable_total as f64 * 100.0
                } else {
                    0.0
                },
                ScoreUnit::Percent,
            ),
        );
        scores.insert("total".to_string(), Score::integer(total, ScoreUnit::Count));
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
                    "SQuAD 2.0: EM {:.1}%, F1 {:.1}% (answerable), unanswerable accuracy {:.1}% ({}/{})",
                    em_score, f1_score, unanswerable_correct as f64 / unanswerable_total.max(1) as f64 * 100.0,
                    unanswerable_correct, unanswerable_total
                ),
            }],
            raw: raw.clone(),
        })
    }
}

fn compute_f1(reference: &str, response: &str) -> f64 {
    let ref_lower = reference.to_lowercase();
    let resp_lower = response.to_lowercase();
    let reference_tokens: Vec<&str> = ref_lower.split_whitespace().collect();
    let response_tokens: Vec<&str> = resp_lower.split_whitespace().collect();

    let intersection: usize = reference_tokens
        .iter()
        .filter(|r| response_tokens.contains(r))
        .count();

    let precision = if response_tokens.is_empty() {
        0.0
    } else {
        intersection as f64 / response_tokens.len() as f64
    };
    let recall = if reference_tokens.is_empty() {
        0.0
    } else {
        intersection as f64 / reference_tokens.len() as f64
    };

    if precision + recall == 0.0 {
        0.0
    } else {
        2.0 * precision * recall / (precision + recall)
    }
}
