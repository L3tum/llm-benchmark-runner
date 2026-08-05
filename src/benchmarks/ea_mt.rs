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

pub struct EAMTBenchmark {
    state: Mutex<EAMTState>,
}

struct EAMTState {
    items: Vec<EAItem>,
    current_idx: usize,
}

impl Default for EAMTBenchmark {
    fn default() -> Self {
        Self {
            state: Mutex::new(EAMTState {
                items: Vec::new(),
                current_idx: 0,
            }),
        }
    }
}

#[derive(Debug, Clone, Deserialize)]
#[allow(dead_code)] // entities field kept for schema alignment
struct EAItem {
    sentence_id: String,
    source_language: String,
    target_language: String,
    sentence: String, // source sentence
    target: String,   // reference translation
    entities: Vec<Entity>,
}

#[derive(Debug, Clone, Deserialize)]
#[allow(dead_code)] // entity fields kept for schema alignment
struct Entity {
    translation: String,
    #[serde(default)]
    mention: String,
}

/// Raw schema of the current EA-MT per-language JSONL files.
#[derive(Debug, Deserialize)]
struct RawEAItem {
    id: String,
    #[serde(rename = "source_locale")]
    source_language: String,
    #[serde(rename = "target_locale")]
    target_language: String,
    source: String,
    targets: Vec<RawTarget>,
}

#[derive(Debug, Deserialize)]
struct RawTarget {
    translation: String,
    #[serde(default)]
    mention: String,
}

impl From<RawEAItem> for EAItem {
    fn from(r: RawEAItem) -> Self {
        let entities: Vec<Entity> = r
            .targets
            .into_iter()
            .map(|t| Entity {
                translation: t.translation,
                mention: t.mention,
            })
            .collect();
        let target = entities
            .first()
            .map(|e| e.translation.clone())
            .unwrap_or_default();
        EAItem {
            sentence_id: r.id,
            source_language: r.source_language,
            target_language: r.target_language,
            sentence: r.source,
            target,
            entities,
        }
    }
}

/// EA-MT test languages available in the HuggingFace repo (data/test/<locale>.jsonl).
const EA_MT_TEST_LOCALES: &[&str] = &[
    "ar_AE", "de_DE", "es_ES", "fr_FR", "it_IT", "ja_JP", "ko_KR", "th_TH", "tr_TR", "zh_TW",
];

fn load_eamt_dataset() -> Vec<EAItem> {
    let cache_dir = dirs::cache_dir()
        .unwrap_or_default()
        .join("llm-benchmark-runner")
        .join("ea_mt");
    fs::create_dir_all(&cache_dir).expect("Failed to create cache dir");

    let mut items = Vec::new();
    for locale in EA_MT_TEST_LOCALES {
        let path = cache_dir.join(format!("test-{}.jsonl", locale));
        if !path.exists() {
            println!("  Downloading EA-MT test ({})...", locale);
            let url = format!(
                "https://huggingface.co/datasets/sapienzanlp/ea-mt-benchmark/resolve/main/data/test/{}.jsonl",
                locale
            );
            let bytes = download_with_retry_bytes(&url, 3, 60, "llm-benchmark-runner")
                .expect("Failed to download EA-MT");
            fs::write(&path, &bytes).expect("Failed to save EA-MT");
        }
        let content = fs::read_to_string(&path).expect("Failed to read cached EA-MT");
        for line in content.lines() {
            if line.trim().is_empty() {
                continue;
            }
            let raw: RawEAItem = serde_json::from_str(line).expect("Failed to parse EA-MT line");
            items.push(EAItem::from(raw));
        }
    }
    items
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
        let items = load_eamt_dataset();
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
            "You are a translation expert. Translate the given sentence into the target language. Pay special attention to named entities — translate them correctly if appropriate.";

        let user_prompt = "Translate from {source_language} to {target_language}:\n\nSource: {sentence}\nTranslation:";
        let prompt = user_prompt
            .replace("{source_language}", &item.source_language)
            .replace("{target_language}", &item.target_language)
            .replace("{sentence}", &item.sentence);

        let response = tracker.chat_completion(&model.model_name, system_prompt, &prompt)?;

        let response = response.trim().to_lowercase();
        let reference = item.target.trim().to_lowercase();
        let is_correct = response == reference;

        let lang_pair = format!("{}-{}", item.source_language, item.target_language);
        let categories = vec![lang_pair];

        Ok(Some(
            TaskResult::new(
                format!("task-{}", idx),
                is_correct,
                if is_correct { 1.0 } else { 0.0 },
                categories,
            )
            .with_metadata(Some(serde_json::json!({
                "sentence_id": item.sentence_id,
                "reference": item.target,
                "response": response,
            }))),
        ))
    }

    fn to_report_result(&self, b: &BenchmarkResult) -> Result<BenchmarkResult> {
        let raw = &b.raw;

        let (total, correct, output_tokens, thinking_tokens, pair_stats) = {
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

                let mut pair_stats: BTreeMap<String, (i64, i64)> = BTreeMap::new();
                for task in per_task {
                    if let Some(pair) = task
                        .get("categories")
                        .and_then(|v| v.as_array())
                        .and_then(|arr| arr.first())
                        .and_then(|v| v.as_str())
                    {
                        let (p, t) = pair_stats.entry(pair.to_string()).or_insert((0, 0));
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
                (total, correct, out, think, pair_stats)
            } else {
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

        // Language pair breakdown
        let mut breakdowns = BTreeMap::new();
        if !pair_stats.is_empty() {
            let mut rows = BTreeMap::new();
            for (pair, (pair_correct, pair_total)) in &pair_stats {
                let rate = if *pair_total > 0 {
                    *pair_correct as f64 / *pair_total as f64
                } else {
                    0.0
                };
                rows.insert(
                    pair.clone(),
                    BTreeMap::from([
                        (
                            "accuracy".to_string(),
                            Score::float(rate * 100.0, ScoreUnit::Percent)
                                .display(format!("{:.1}%", rate * 100.0)),
                        ),
                        (
                            "instances".to_string(),
                            Score::integer(*pair_total, ScoreUnit::Count)
                                .display(format!("{}/{}", pair_correct, pair_total)),
                        ),
                    ]),
                );
            }
            breakdowns.insert(
                "By Language Pair".to_string(),
                crate::reports::model::BreakdownTable {
                    title: "Accuracy by Language Pair".to_string(),
                    rows,
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
