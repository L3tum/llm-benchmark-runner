use crate::benchmarks::Benchmark;
use crate::config::Model;
use crate::download::download_parquet_records;
use crate::shared::{BenchmarkCategory, BenchmarkResult, Score, ScoreUnit, TaskResult};
use crate::token_tracker::TokenTracker;
use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::fs;
use std::sync::Mutex;

pub struct MmluProxBenchmark {
    state: Mutex<MmluProXState>,
}

#[derive(Clone, serde::Serialize, serde::Deserialize)]
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

#[derive(Debug, Clone, Serialize, Deserialize)]
#[allow(dead_code)] // subject field kept for schema alignment
struct MmluProXItem {
    id: String,
    language: String,
    category: String,
    question: String,
    choices: Vec<String>,
    correct_answer: String, // single letter
    subject: String,
}

const ALL_LANGUAGES: &[&str] = &[
    "af", "ar", "bn", "cs", "de", "en", "es", "fr", "hi", "hu", "id", "it", "ja", "ko", "mr", "ne",
    "pt", "ru", "sr", "sw", "te", "th", "uk", "ur", "vi", "wo", "yo", "zh", "zu",
];

fn load_language(cache_dir: &std::path::Path, lang: &str) -> Result<Vec<MmluProXItem>> {
    let path = cache_dir.join(format!("{}.json", lang));
    if path.exists() {
        let content = fs::read_to_string(&path)?;
        return serde_json::from_str(&content).context("parse cached MMLU-ProX");
    }
    let url = format!(
        "https://huggingface.co/datasets/li-lab/MMLU-ProX/resolve/main/{}/test-00000-of-00001.parquet",
        lang
    );
    println!("  Downloading MMLU-ProX language '{}'...", lang);
    let rows = download_parquet_records(&url, 3, 60, "llm-benchmark-runner")
        .with_context(|| format!("download MMLU-ProX {} parquet", lang))?;
    let items: Vec<MmluProXItem> = rows
        .iter()
        .map(|r| {
            let mut choices = Vec::new();
            for i in 0..10 {
                if let Some(v) = r.get(format!("option_{}", i)).and_then(|v| v.as_str()) {
                    choices.push(v.to_string());
                }
            }
            MmluProXItem {
                id: r
                    .get("question_id")
                    .map(|v| v.to_string())
                    .unwrap_or_default(),
                language: lang.to_string(),
                category: r
                    .get("category")
                    .and_then(|v| v.as_str())
                    .unwrap_or("")
                    .to_string(),
                question: r
                    .get("question")
                    .and_then(|v| v.as_str())
                    .unwrap_or("")
                    .to_string(),
                choices,
                correct_answer: r
                    .get("answer")
                    .and_then(|v| v.as_str())
                    .unwrap_or("")
                    .to_string(),
                subject: r
                    .get("src")
                    .and_then(|v| v.as_str())
                    .unwrap_or("")
                    .to_string(),
            }
        })
        .collect();
    fs::write(&path, serde_json::to_vec(&items)?).context("save MMLU-ProX cache")?;
    Ok(items)
}

fn load_mmlu_prox(languages: &[String]) -> Result<Vec<MmluProXItem>> {
    let cache_dir = dirs::cache_dir()
        .unwrap_or_default()
        .join("llm-benchmark-runner")
        .join("mmlu_prox");
    fs::create_dir_all(&cache_dir)?;
    let mut all = Vec::new();
    for lang in languages {
        let lang = lang.trim().to_lowercase();
        if lang.is_empty() {
            continue;
        }
        match load_language(&cache_dir, &lang) {
            Ok(items) => all.extend(items),
            Err(e) => eprintln!("  WARNING: skipping MMLU-ProX language '{}': {}", lang, e),
        }
    }
    Ok(all)
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

    fn pre_execute(&self, config: &yaml_serde::Value) -> Result<()> {
        let languages: Vec<String> = crate::config::extract_string_vec(config, "languages")
            .filter(|v| !v.is_empty())
            .unwrap_or_else(|| ALL_LANGUAGES.iter().map(|s| s.to_string()).collect());
        let items = load_mmlu_prox(&languages)?;
        println!(
            "MMLU-ProX: {} questions across {} language(s): {}",
            items.len(),
            languages.len(),
            languages.join(", ")
        );
        let mut state = self.state.lock().unwrap_or_else(|p| p.into_inner());
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
            let mut state = self.state.lock().unwrap_or_else(|p| p.into_inner());
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
