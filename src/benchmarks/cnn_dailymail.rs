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

/// CNN/Daily Mail benchmark with ROUGE-based faithfulness proxy.
///
/// NOTE: The XSum Faithfulness dataset (EdinburghNLP/xsum_faithfulness) provides human-annotated
/// hallucination spans. Since it is gated, we use ROUGE overlap with the human-written highlights
/// as a faithfulness proxy — a summary that closely matches the faithful reference is likely faithful.
pub struct CnnDailyMailBenchmark;

static DATASET: OnceLock<Vec<CnnDmItem>> = OnceLock::new();

#[derive(Debug, Clone, Deserialize)]
struct CnnDmItem {
    id: String,
    article: String,
    highlights: String,
}

fn load_cnn_dailymail() -> &'static Vec<CnnDmItem> {
    DATASET.get_or_init(|| {
        let cache_dir = dirs::cache_dir()
            .unwrap_or_default()
            .join("llm-benchmark-runner")
            .join("cnn_dailymail");
        let path = cache_dir.join("cnn_dailymail.json");

        if path.exists() {
            let content = fs::read_to_string(&path).expect("Failed to read cached CNN/DM");
            return serde_json::from_str(&content).expect("Failed to parse CNN/DM");
        }

        fs::create_dir_all(&cache_dir).expect("Failed to create cache dir");
        println!("  Downloading CNN/Daily Mail dataset...");
        let url =
            "https://huggingface.co/datasets/EdinburghNLP/cnn_dailymail/resolve/main/test.csv";
        let bytes = download_with_retry_bytes(url, 3, 60, "llm-benchmark-runner")
            .expect("Failed to download CNN/DM");

        // Parse CSV (3 columns: id, article, highlights)
        let content = String::from_utf8(Vec::from(bytes.as_ref())).expect("Failed to decode UTF-8");
        let mut items = Vec::new();
        for line in content.lines().skip(1) {
            if line.is_empty() {
                continue;
            }
            let fields: Vec<&str> = line.split(",").collect();
            if fields.len() >= 3 {
                items.push(CnnDmItem {
                    id: fields[0].trim_matches('"').to_string(),
                    article: fields[1].trim_matches('"').to_string(),
                    highlights: fields[2].trim_matches('"').to_string(),
                });
            }
        }

        fs::write(&path, &bytes).expect("Failed to save CNN/DM");
        items
    })
}

impl Benchmark for CnnDailyMailBenchmark {
    fn name(&self) -> &str {
        "cnn_dailymail"
    }

    fn display_name(&self) -> &'static str {
        "CNN/Daily Mail (Faithfulness Proxy)"
    }

    fn category(&self) -> BenchmarkCategory {
        BenchmarkCategory::Hallucination
    }

    fn pre_execute(&self, _config: &yaml_serde::Value) -> Result<()> {
        let _ = load_cnn_dailymail();
        Ok(())
    }

    fn execute(&self, model: &Model, _config: &yaml_serde::Value) -> Result<BenchmarkResult> {
        let dataset = load_cnn_dailymail();
        let client = Client::new_with_model_params(&model.proxy, model.set_params.as_ref())?;

        let system_prompt =
            "You are a summarisation expert. Given a news article, write a few-sentence summary (3-5 sentences) that captures the key points. Do not include any information not present in the article.";

        let user_prompt = r#"Article: {article}
Summary:"#;

        let total = dataset.len();
        let mut rouge1_score = 0.0;
        let mut rouge2_score = 0.0;
        let mut rouge_l_score = 0.0;
        let mut output_tokens_total: i64 = 0;
        let mut thinking_tokens_total: i64 = 0;

        for item in dataset {
            let prompt = user_prompt.replace("{article}", &item.article);
            let (response, output_tokens, thinking_tokens) =
                client.chat_completion(&model.model_name, system_prompt, &prompt)?;

            output_tokens_total += output_tokens.unwrap_or(0) as i64;
            thinking_tokens_total += thinking_tokens.unwrap_or(0) as i64;

            let reference = item.highlights.trim();
            let prediction = response.trim();
            let (r1, r2, r_l) = compute_rouge_scores(reference, prediction);
            rouge1_score += r1;
            rouge2_score += r2;
            rouge_l_score += r_l;
        }

        let rouge1 = rouge1_score / total as f64;
        let rouge2 = rouge2_score / total as f64;
        let rouge_l = rouge_l_score / total as f64;

        let raw_json = serde_json::json!({
            "rouge1": rouge1,
            "rouge2": rouge2,
            "rouge_l": rouge_l,
            "total": total,
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
                    "CNN/DM (Faithfulness Proxy): ROUGE-1 {:.1}%, ROUGE-2 {:.1}%, ROUGE-L {:.1}% — overlap with human-annotated faithful highlights",
                    rouge1 * 100.0,
                    rouge2 * 100.0,
                    rouge_l * 100.0
                ),
            }],
            raw: raw_json,
        })
    }

    fn to_report_result(&self, b: &BenchmarkResult) -> Result<BenchmarkResult> {
        let raw = &b.raw;
        let rouge1 = raw.get("rouge1").and_then(|v| v.as_f64()).unwrap_or(0.0);
        let rouge2 = raw.get("rouge2").and_then(|v| v.as_f64()).unwrap_or(0.0);
        let rouge_l = raw.get("rouge_l").and_then(|v| v.as_f64()).unwrap_or(0.0);
        let total = raw.get("total").and_then(|v| v.as_i64()).unwrap_or(0);
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
            "rouge1".to_string(),
            Score::float(rouge1 * 100.0, ScoreUnit::Percent)
                .primary(true)
                .higher_is_better(true),
        );
        scores.insert(
            "rouge2".to_string(),
            Score::float(rouge2 * 100.0, ScoreUnit::Percent),
        );
        scores.insert(
            "rouge_l".to_string(),
            Score::float(rouge_l * 100.0, ScoreUnit::Percent),
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
                    "CNN/DM (Faithfulness Proxy): ROUGE-1 {:.1}%, ROUGE-2 {:.1}%, ROUGE-L {:.1}% — overlap with human-annotated faithful highlights",
                    rouge1 * 100.0,
                    rouge2 * 100.0,
                    rouge_l * 100.0
                ),
            }],
            raw: raw.clone(),
        })
    }
}

fn compute_rouge_scores(reference: &str, prediction: &str) -> (f64, f64, f64) {
    let ref_lower = reference.to_lowercase();
    let pred_lower = prediction.to_lowercase();
    let ref_tokens: Vec<&str> = ref_lower.split_whitespace().collect();
    let pred_tokens: Vec<&str> = pred_lower.split_whitespace().collect();

    let r1 = compute_rouge_n(&ref_tokens, &pred_tokens, 1);
    let r2 = compute_rouge_n(&ref_tokens, &pred_tokens, 2);
    let r_l = compute_rouge_l(&ref_tokens, &pred_tokens);

    (r1, r2, r_l)
}

fn compute_rouge_n(reference: &[&str], prediction: &[&str], n: usize) -> f64 {
    if reference.is_empty() || prediction.is_empty() {
        return 0.0;
    }

    let ref_ngrams = ngrams(reference, n);
    let pred_ngrams = ngrams(prediction, n);

    let common: usize = ref_ngrams
        .iter()
        .map(|ngram| {
            let mut count = 0;
            for pred_ngram in &pred_ngrams {
                if ngram == pred_ngram {
                    count += 1;
                }
            }
            count
        })
        .sum();

    let precision = common as f64 / pred_ngrams.len() as f64;
    let recall = common as f64 / ref_ngrams.len() as f64;

    if precision + recall == 0.0 {
        0.0
    } else {
        2.0 * precision * recall / (precision + recall)
    }
}

fn ngrams<'a>(tokens: &'a [&'a str], n: usize) -> Vec<Vec<&'a str>> {
    if tokens.len() < n {
        return vec![tokens.to_vec()];
    }
    tokens.windows(n).map(|w| w.to_vec()).collect()
}

fn compute_rouge_l(reference: &[&str], prediction: &[&str]) -> f64 {
    let lcs_len = longest_common_subsequence_len(reference, prediction);
    let precision = lcs_len as f64 / prediction.len() as f64;
    let recall = lcs_len as f64 / reference.len() as f64;

    if precision + recall == 0.0 {
        0.0
    } else {
        2.0 * precision * recall / (precision + recall)
    }
}

fn longest_common_subsequence_len(a: &[&str], b: &[&str]) -> usize {
    let m = a.len();
    let n = b.len();
    let mut dp = vec![vec![0usize; n + 1]; m + 1];

    for i in 1..=m {
        for j in 1..=n {
            if a[i - 1] == b[j - 1] {
                dp[i][j] = dp[i - 1][j - 1] + 1;
            } else {
                dp[i][j] = dp[i - 1][j].max(dp[i][j - 1]);
            }
        }
    }
    dp[m][n]
}
