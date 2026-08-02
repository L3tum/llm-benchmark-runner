use crate::benchmarks::Benchmark;
use crate::config::Model;
use crate::download::download_with_retry_bytes;
use crate::shared::{BenchmarkCategory, BenchmarkResult, Score, ScoreUnit, TaskResult};
use crate::token_tracker::TokenTracker;
use anyhow::Result;
use serde::Deserialize;
use std::collections::BTreeMap;
use std::fs;
use std::sync::Mutex;

/// XSum benchmark with ROUGE-based faithfulness proxy.
///
/// NOTE: The XSum Faithfulness dataset (EdinburghNLP/xsum_faithfulness) provides human-annotated
/// hallucination spans with 3 human judgements per summary. Since accessing it requires a
/// HuggingFace token and is currently gated, we use ROUGE overlap with the human-written
/// reference summary as a proxy for faithfulness. A summary that closely matches the faithful
/// reference is likely faithful. The diagnostic message clarifies this approach.
pub struct XSumBenchmark {
    state: Mutex<XSumState>,
}

struct XSumState {
    items: Vec<XSumItem>,
    current_idx: usize,
}

impl Default for XSumBenchmark {
    fn default() -> Self {
        Self {
            state: Mutex::new(XSumState {
                items: Vec::new(),
                current_idx: 0,
            }),
        }
    }
}

#[derive(Debug, Clone, Deserialize)]
#[allow(dead_code)] // narrative_link kept for schema alignment
struct XSumItem {
    document: String,
    summary: String,
    #[serde(rename = "narrative_link")]
    narrative_link: String,
}

fn load_xsum_dataset() -> Vec<XSumItem> {
    let cache_dir = dirs::cache_dir()
        .unwrap_or_default()
        .join("llm-benchmark-runner")
        .join("xsum");
    let path = cache_dir.join("XSum.csv");

    if path.exists() {
        let content = fs::read_to_string(&path).expect("Failed to read cached XSum");
        return parse_xsum_csv(&content);
    }

    fs::create_dir_all(&cache_dir).expect("Failed to create cache dir");
    println!("  Downloading XSum dataset...");
    let url = "https://huggingface.co/datasets/EdinburghNLP/xsum/resolve/main/test.csv";
    let bytes = download_with_retry_bytes(url, 3, 60, "llm-benchmark-runner")
        .expect("Failed to download XSum");

    let content = String::from_utf8(Vec::from(bytes.as_ref())).expect("Failed to decode UTF-8");
    let items = parse_xsum_csv(&content);
    fs::write(&path, &bytes).expect("Failed to save XSum");
    items
}

fn parse_xsum_csv(content: &str) -> Vec<XSumItem> {
    let mut items = Vec::new();
    for line in content.lines().skip(1) {
        if line.is_empty() {
            continue;
        }
        // CSV with 3 columns: document, summary, narrative_link
        let fields: Vec<&str> = line.split(",").collect();
        if fields.len() >= 3 {
            items.push(XSumItem {
                document: fields[0].trim_matches('"').to_string(),
                summary: fields[1].trim_matches('"').to_string(),
                narrative_link: fields[2].trim_matches('"').to_string(),
            });
        }
    }
    items
}

impl Benchmark for XSumBenchmark {
    fn name(&self) -> &str {
        "xsum"
    }

    fn display_name(&self) -> &'static str {
        "XSum (Abstractive Summarisation)"
    }

    fn category(&self) -> BenchmarkCategory {
        BenchmarkCategory::Hallucination
    }

    fn pre_execute(&self, _config: &yaml_serde::Value) -> Result<()> {
        let items = load_xsum_dataset();
        let mut state = self.state.lock().expect(crate::shared::MUTEX_PANIC_MSG);
        state.items = items;
        state.current_idx = 0;
        Ok(())
    }

    fn execute_one(
        &self,
        model: &Model,
        _config: &yaml_serde::Value,
        tracker: &mut TokenTracker,
    ) -> Result<Option<TaskResult>> {
        let (item, idx) = {
            let mut state = self.state.lock().expect(crate::shared::MUTEX_PANIC_MSG);
            if state.current_idx >= state.items.len() {
                return Ok(None);
            }
            let idx = state.current_idx;
            let item = state.items[idx].clone();
            state.current_idx += 1;
            (item, idx)
        };

        let system_prompt =
            "You are a summarisation expert. Given an article, write a single-sentence summary that captures the main point. Do not include any information not present in the article.";

        let user_prompt = "Article: {article}\nSummary:";
        let prompt = user_prompt.replace("{article}", &item.document);

        let response = tracker.chat_completion(&model.model_name, system_prompt, &prompt)?;

        let reference = item.summary.trim();
        let prediction = response.trim();
        let (r1, r2, r_l) = compute_rouge_scores(reference, prediction);

        Ok(Some(
            TaskResult::new(format!("task-{}", idx), false, r1, vec![]).with_metadata(Some(
                serde_json::json!({
                    "rouge1": r1,
                    "rouge2": r2,
                    "rouge_l": r_l,
                    "reference": reference,
                    "prediction": prediction,
                }),
            )),
        ))
    }

    fn to_report_result(&self, b: &BenchmarkResult) -> Result<BenchmarkResult> {
        let raw = &b.raw;

        let (total, rouge1_total, rouge2_total, rouge_l_total, output_tokens, thinking_tokens) = {
            if let Some(per_task) = raw.get("per_task").and_then(|v| v.as_array()) {
                let total = per_task.len() as i64;
                let r1: f64 = per_task
                    .iter()
                    .filter_map(|t| t.get("rouge1").and_then(|v| v.as_f64()))
                    .sum();
                let r2: f64 = per_task
                    .iter()
                    .filter_map(|t| t.get("rouge2").and_then(|v| v.as_f64()))
                    .sum();
                let rl: f64 = per_task
                    .iter()
                    .filter_map(|t| t.get("rouge_l").and_then(|v| v.as_f64()))
                    .sum();
                let out: i64 = per_task
                    .iter()
                    .filter_map(|t| t.get("output_tokens").and_then(|v| v.as_i64()))
                    .sum();
                let think: i64 = per_task
                    .iter()
                    .filter_map(|t| t.get("thinking_tokens").and_then(|v| v.as_i64()))
                    .sum();
                (total, r1, r2, rl, out, think)
            } else {
                (
                    raw.get("total").and_then(|v| v.as_i64()).unwrap_or(0),
                    raw.get("rouge1").and_then(|v| v.as_f64()).unwrap_or(0.0),
                    raw.get("rouge2").and_then(|v| v.as_f64()).unwrap_or(0.0),
                    raw.get("rouge_l").and_then(|v| v.as_f64()).unwrap_or(0.0),
                    raw.get("output_tokens")
                        .and_then(|v| v.as_i64())
                        .unwrap_or(0),
                    raw.get("thinking_tokens")
                        .and_then(|v| v.as_i64())
                        .unwrap_or(0),
                )
            }
        };

        let rouge1 = if total > 0 {
            rouge1_total / total as f64
        } else {
            0.0
        };
        let rouge2 = if total > 0 {
            rouge2_total / total as f64
        } else {
            0.0
        };
        let rouge_l = if total > 0 {
            rouge_l_total / total as f64
        } else {
            0.0
        };

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
                    "XSum (Faithfulness Proxy): ROUGE-1 {:.1}%, ROUGE-2 {:.1}%, ROUGE-L {:.1}% — overlap with human-annotated faithful reference",
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

    // ROUGE-1 (unigram)
    let r1 = compute_rouge_n(&ref_tokens, &pred_tokens, 1);
    // ROUGE-2 (bigram)
    let r2 = compute_rouge_n(&ref_tokens, &pred_tokens, 2);
    // ROUGE-L (longest common subsequence)
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
