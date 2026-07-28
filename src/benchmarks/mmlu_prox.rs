use crate::benchmarks::Benchmark;
use crate::config::Model;
use crate::download::download_with_retry_bytes;
use crate::reports::model::{BenchmarkCategory, BenchmarkResult, Score, ScoreUnit, TaskResult};
use crate::token_tracker::TokenTracker;
use anyhow::Result;
use serde::Deserialize;
use std::collections::BTreeMap;
use std::fs;
use std::sync::Mutex;

pub struct MmluProxBenchmark {
    state: Mutex<MmluProXState>,
}

struct MmluProXState {
    items: Vec<MmluProXItem>,
    current_idx: usize,
}

impl Default for MmluProxBenchmark {
    fn default() -> Self {
        Self {
            state: Mutex::new(MmluProXState {
                items: Vec::new(),
                current_idx: 0,
            }),
        }
    }
}

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

fn load_mmlu_prox() -> Vec<MmluProXItem> {
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
        let items = load_mmlu_prox();
        let mut state = self.state.lock().unwrap();
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
            let mut state = self.state.lock().unwrap();
            if state.current_idx >= state.items.len() {
                return Ok(None);
            }
            let idx = state.current_idx;
            let item = state.items[idx].clone();
            state.current_idx += 1;
            (item, idx)
        };

        let system_prompt =
            "You are a multilingual multiple-choice question expert. Select the single correct answer from the options. Respond with only the letter.";

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

        let mut choices_str = String::new();
        let labels: Vec<char> = ('A'..='J').collect();
        for (i, choice) in item.choices.iter().enumerate() {
            choices_str.push_str(&format!("{}. {}\n", labels[i], choice));
        }

        let prompt = user_prompt
            .replace("{question}", &item.question)
            .replace("{choices}", &choices_str);

        let response = tracker.chat_completion(&model.model_name, system_prompt, &prompt)?;

        let response_letter = response.chars().next().unwrap_or('Z').to_ascii_uppercase();
        let expected = item
            .correct_answer
            .chars()
            .next()
            .unwrap_or('A')
            .to_ascii_uppercase();
        let is_correct = response_letter == expected;

        let categories = vec![item.language.clone()];

        Ok(Some(
            TaskResult::new(
                format!("task-{}", idx),
                is_correct,
                if is_correct { 1.0 } else { 0.0 },
                categories,
            )
            .with_metadata(Some(serde_json::json!({
                "id": item.id,
                "language": item.language,
                "category": item.category,
                "expected": expected,
                "response": response_letter,
            }))),
        ))
    }

    fn to_report_result(&self, b: &BenchmarkResult) -> Result<BenchmarkResult> {
        let raw = &b.raw;

        // Read from per_task array
        let (total, correct, output_tokens, thinking_tokens, lang_stats) = {
            if let Some(per_task) = raw.get("per_task").and_then(|v| v.as_array()) {
                let total = per_task.len() as i64;
                let correct = per_task
                    .iter()
                    .filter(|t| t.get("passed").and_then(|v| v.as_bool()).unwrap_or(false))
                    .count() as i64;
                let out: i64 = per_task
                    .iter()
                    .filter_map(|t| t.get("output_tokens").and_then(|v| v.as_i64()))
                    .sum();
                let think: i64 = per_task
                    .iter()
                    .filter_map(|t| t.get("thinking_tokens").and_then(|v| v.as_i64()))
                    .sum();

                let mut lang_stats: BTreeMap<String, (i64, i64)> = BTreeMap::new();
                for task in per_task {
                    if let Some(lang) = task
                        .get("categories")
                        .and_then(|v| v.as_array())
                        .and_then(|arr| arr.first())
                        .and_then(|v| v.as_str())
                    {
                        let (p, t) = lang_stats.entry(lang.to_string()).or_insert((0, 0));
                        *t += 1;
                        if task
                            .get("passed")
                            .and_then(|v| v.as_bool())
                            .unwrap_or(false)
                        {
                            *p += 1;
                        }
                    }
                }
                (total, correct, out, think, lang_stats)
            } else {
                // Fallback for deserialized results
                (
                    raw.get("total").and_then(|v| v.as_i64()).unwrap_or(0),
                    raw.get("correct").and_then(|v| v.as_i64()).unwrap_or(0),
                    raw.get("output_tokens")
                        .and_then(|v| v.as_i64())
                        .unwrap_or(0),
                    raw.get("thinking_tokens")
                        .and_then(|v| v.as_i64())
                        .unwrap_or(0),
                    BTreeMap::new(),
                )
            }
        };

        let accuracy = if total > 0 {
            correct as f64 / total as f64
        } else {
            0.0
        };

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

        // Language breakdown
        let mut breakdowns = BTreeMap::new();
        for (lang, (lang_correct, lang_total)) in &lang_stats {
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
                                    *lang_correct as f64 / *lang_total as f64 * 100.0,
                                    ScoreUnit::Percent,
                                ),
                            )]),
                        ),
                        (
                            "count".to_string(),
                            BTreeMap::from_iter([(
                                "total".to_string(),
                                Score::integer(*lang_total, ScoreUnit::Count),
                            )]),
                        ),
                    ]),
                },
            );
        }

        Ok(BenchmarkResult {
            scores,
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
            raw: raw.clone(),
        })
    }
}
