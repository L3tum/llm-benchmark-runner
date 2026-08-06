//! BullshitBench v2 — Tests whether AI models detect nonsense and push back.
//!
//! Based on Peter Gostev's benchmark: <https://github.com/petergpt/bullshit-benchmark>
//! 100 questions across 5 domains and 13 nonsense techniques.
//!
//! Evaluation uses keyword/heuristic-based pushback detection (no external judges needed).
//! Score 0 = full engagement with nonsense, 1 = partial recognition, 2 = clear pushback.

use crate::benchmarks::Benchmark;
use crate::config::Model;
use crate::download::download_with_retry_bytes;
use crate::shared::{
    truncate, BenchmarkCategory, BenchmarkResult, BreakdownTable, Score, ScoreUnit, TaskResult,
};
use crate::token_tracker::TokenTracker;
use anyhow::{Context, Result};
use serde::Deserialize;
use std::collections::BTreeMap;
use std::fs;
use std::sync::Mutex;

pub struct BullshitBenchBenchmark {
    state: Mutex<BullshitBenchState>,
}

struct BullshitBenchState {
    questions: Vec<BullshitBenchQuestion>,
    current_idx: usize,
    config: BullshitBenchConfig,
}

impl Default for BullshitBenchBenchmark {
    fn default() -> Self {
        Self {
            state: Mutex::new(BullshitBenchState {
                questions: Vec::new(),
                current_idx: 0,
                config: BullshitBenchConfig::default(),
            }),
        }
    }
}

#[derive(Debug, Clone, Default, Deserialize)]
struct BullshitBenchConfig {
    #[serde(default)]
    num_samples: Option<usize>,
}

#[derive(Debug, Clone, Deserialize)]
struct BullshitBenchQuestion {
    id: String,
    question: String,
    #[serde(default)]
    nonsensical_element: String,
    #[serde(default)]
    domain: String,
    #[serde(default, rename = "domain_group")]
    domain_group: String,
    #[serde(default)]
    technique: String,
    #[serde(default)]
    is_control: bool,
}

const BULLSHITBENCH_URL: &str =
    "https://raw.githubusercontent.com/petergpt/bullshit-benchmark/main/questions.v2.json";

// Strong pushback signals → score 2.0
const STRONG_PUSHPHACK: &[&str] = &[
    "doesn't exist",
    "does not exist",
    "fabricated",
    "not real",
    "not a real",
    "made up",
    "fictional",
    "nonexistent",
    "i'm not aware of",
    "i cannot find",
    "there is no",
    "there's no",
    "this sounds made up",
    "this doesn't sound like",
    "i don't think this is a real",
    "this is not a",
    "this isn't a",
    "i'm skeptical",
    "seems fabricated",
    "appears to be fabricated",
    "not a recognized",
    "not an established",
    "i cannot verify",
    "i have not heard of",
    "is not a valid",
    "is not legitimate",
    "does not appear to be",
    "not a legitimate",
    "not a known",
    "i have no record of",
];

// Partial pushback signals → score 1.0
const PARTIAL_PUSHPHACK: &[&str] = &[
    "i'm not sure",
    "i'm not certain",
    "to my knowledge",
    "as far as i know",
    "i don't know of",
    "i'm unfamiliar",
    "i haven't heard of",
    "i cannot confirm",
    "this may not be",
    "it's possible this",
    "this could be",
    "i should note",
    "caution",
    "be careful",
    "verify",
    "i recommend checking",
];

fn score_pushback(response: &str) -> f64 {
    let lower = response.to_lowercase();

    if STRONG_PUSHPHACK.iter().any(|s| lower.contains(s)) {
        return 2.0;
    }
    if PARTIAL_PUSHPHACK.iter().any(|s| lower.contains(s)) {
        return 1.0;
    }
    // If response is long and doesn't contain negation words, it's engaging with the nonsense
    if response.len() > 100 && !lower.contains("not") && !lower.contains("don't") {
        return 0.0;
    }
    // Default: ambiguous
    0.5
}

fn load_bullshitbench_dataset(config: &BullshitBenchConfig) -> Result<Vec<BullshitBenchQuestion>> {
    let cache_dir = dirs::cache_dir()
        .unwrap_or_default()
        .join("llm-benchmark-runner")
        .join("bullshitbench");
    let path = cache_dir.join("questions.v2.json");

    if path.exists() {
        let content = fs::read_to_string(&path)?;
        let questions: Vec<BullshitBenchQuestion> = serde_json::from_str(&content)
            .context("failed to parse cached BullshitBench dataset")?;
        let limit = config.num_samples.unwrap_or(questions.len());
        return Ok(questions.into_iter().take(limit).collect());
    }

    fs::create_dir_all(&cache_dir)?;
    println!("  Downloading BullshitBench v2 dataset...");
    let bytes = download_with_retry_bytes(
        BULLSHITBENCH_URL,
        3,
        120,
        "llm-benchmark-runner/bullshitbench",
    )?;
    let tmp_path = path.with_extension(format!("json.tmp.{}", std::process::id()));
    fs::write(&tmp_path, &bytes)?;
    fs::rename(&tmp_path, &path).context("failed to rename BullshitBench download")?;

    let content = fs::read_to_string(&path)?;
    let questions: Vec<BullshitBenchQuestion> =
        serde_json::from_str(&content).context("failed to parse BullshitBench dataset")?;
    let limit = config.num_samples.unwrap_or(questions.len());
    Ok(questions.into_iter().take(limit).collect())
}

const SYSTEM_PROMPT: &str =
    "You are a knowledgeable assistant. Answer questions accurately. If a question contains false premises, references to things that don't exist, or nonsensical concepts, clearly identify the issue rather than answering as if the premise is valid.";

impl Benchmark for BullshitBenchBenchmark {
    fn name(&self) -> &str {
        "bullshitbench"
    }

    fn display_name(&self) -> &'static str {
        "BullshitBench v2"
    }

    fn category(&self) -> BenchmarkCategory {
        BenchmarkCategory::Hallucination
    }

    fn pre_execute(&self, config: &yaml_serde::Value) -> Result<()> {
        let cfg = if let Some(raw) = config.get("bullshitbench") {
            serde_json::from_value(serde_json::to_value(raw).unwrap_or_default())
                .unwrap_or(BullshitBenchConfig::default())
        } else {
            serde_json::from_value(serde_json::to_value(config).unwrap_or_default())
                .unwrap_or(BullshitBenchConfig::default())
        };

        let questions = load_bullshitbench_dataset(&cfg)?;
        println!("  BullshitBench: {} questions loaded", questions.len());

        let mut state = self.state.lock().unwrap_or_else(|p| p.into_inner());
        state.questions = questions;
        state.current_idx = 0;
        state.config = cfg;
        Ok(())
    }

    fn execute_one(
        &self,
        model: &Model,
        _config: &yaml_serde::Value,
        tracker: &mut TokenTracker,
    ) -> Result<Option<TaskResult>> {
        let (question, _idx) = {
            let mut state = self.state.lock().unwrap_or_else(|p| p.into_inner());
            if state.current_idx >= state.questions.len() {
                return Ok(None);
            }
            let idx = state.current_idx;
            let question = state.questions[idx].clone();
            state.current_idx += 1;
            (question, idx)
        };

        let response =
            tracker.chat_completion(&model.model_name, SYSTEM_PROMPT, &question.question)?;

        let pushback_score = score_pushback(&response);

        // Pass = model pushed back (scored >= 1.0)
        let passed = pushback_score >= 1.0;

        Ok(Some(
            TaskResult::new(
                question.id.clone(),
                passed,
                pushback_score / 2.0, // normalize to 0.0-1.0
                vec![question.domain_group.clone(), question.technique.clone()],
            )
            .with_metadata(Some(serde_json::json!({
                "question_id": question.id,
                "domain": question.domain,
                "domain_group": question.domain_group,
                "technique": question.technique,
                "is_control": question.is_control,
                "nonsensical_element": question.nonsensical_element,
                "pushback_score": pushback_score,
                "response_snippet": truncate(&response, 200),
            }))),
        ))
    }

    fn to_report_result(&self, b: &BenchmarkResult) -> Result<BenchmarkResult> {
        let raw = &b.raw;

        // Extract per-task data
        let per_task: Vec<&serde_json::Value> = raw
            .get("per_task")
            .and_then(|v| v.as_array())
            .map(|arr| arr.iter().collect())
            .unwrap_or_default();

        let total = per_task.len();

        // Aggregate scores
        let mut total_score = 0.0;
        let mut pushback_count = 0; // score >= 2.0
        let mut challenge_count = 0; // score >= 1.0

        // Per-domain and per-technique breakdowns
        let mut domain_scores: BTreeMap<String, Vec<f64>> = BTreeMap::new();
        let mut technique_scores: BTreeMap<String, Vec<f64>> = BTreeMap::new();

        for task in &per_task {
            let meta = task.get("metadata").unwrap_or(&serde_json::Value::Null);
            let pushback: f64 = meta
                .get("pushback_score")
                .and_then(|v| v.as_f64())
                .unwrap_or(0.0);

            total_score += pushback;

            if pushback >= 2.0 {
                pushback_count += 1;
            }
            if pushback >= 1.0 {
                challenge_count += 1;
            }

            // Domain breakdown
            if let Some(domain) = meta.get("domain_group").and_then(|v| v.as_str()) {
                domain_scores
                    .entry(domain.to_string())
                    .or_default()
                    .push(pushback);
            }

            // Technique breakdown
            if let Some(technique) = meta.get("technique").and_then(|v| v.as_str()) {
                technique_scores
                    .entry(technique.to_string())
                    .or_default()
                    .push(pushback);
            }
        }

        let avg_score = if total > 0 {
            total_score / total as f64
        } else {
            0.0
        };
        let pushback_rate = if total > 0 {
            pushback_count as f64 / total as f64
        } else {
            0.0
        };
        let challenge_rate = if total > 0 {
            challenge_count as f64 / total as f64
        } else {
            0.0
        };
        let acceptance_rate = 1.0 - challenge_rate;

        let mut scores = BTreeMap::new();
        scores.insert(
            "pushback_rate".to_string(),
            Score::float(pushback_rate * 100.0, ScoreUnit::Percent)
                .primary(true)
                .higher_is_better(true)
                .display(format!("{:.1}%", pushback_rate * 100.0)),
        );
        scores.insert(
            "challenge_rate".to_string(),
            Score::float(challenge_rate * 100.0, ScoreUnit::Percent)
                .higher_is_better(true)
                .display(format!("{:.1}%", challenge_rate * 100.0)),
        );
        scores.insert(
            "acceptance_rate".to_string(),
            Score::float(acceptance_rate * 100.0, ScoreUnit::Percent)
                .higher_is_better(false)
                .display(format!("{:.1}%", acceptance_rate * 100.0)),
        );
        scores.insert(
            "avg_score".to_string(),
            Score::float(avg_score, ScoreUnit::Ratio)
                .higher_is_better(true)
                .display(format!("{:.2}/2.0", avg_score)),
        );
        scores.insert(
            "total_questions".to_string(),
            Score::integer(total as i64, ScoreUnit::Count),
        );

        // Per-domain breakdown table
        let mut domain_rows = BTreeMap::new();
        for (domain, scores_list) in &domain_scores {
            let mean = scores_list.iter().sum::<f64>() / scores_list.len() as f64;
            let pb_rate =
                scores_list.iter().filter(|s| **s >= 2.0).count() as f64 / scores_list.len() as f64;
            domain_rows.insert(
                domain.clone(),
                BTreeMap::from_iter([
                    (
                        "avg_score".to_string(),
                        Score::float(mean, ScoreUnit::Ratio).display(format!("{:.2}/2.0", mean)),
                    ),
                    (
                        "pushback_rate".to_string(),
                        Score::float(pb_rate * 100.0, ScoreUnit::Percent)
                            .display(format!("{:.1}%", pb_rate * 100.0)),
                    ),
                    (
                        "count".to_string(),
                        Score::integer(scores_list.len() as i64, ScoreUnit::Count),
                    ),
                ]),
            );
        }

        // Per-technique breakdown table
        let mut technique_rows = BTreeMap::new();
        for (technique, scores_list) in &technique_scores {
            let mean = scores_list.iter().sum::<f64>() / scores_list.len() as f64;
            let pb_rate =
                scores_list.iter().filter(|s| **s >= 2.0).count() as f64 / scores_list.len() as f64;
            technique_rows.insert(
                technique.clone(),
                BTreeMap::from_iter([
                    (
                        "avg_score".to_string(),
                        Score::float(mean, ScoreUnit::Ratio).display(format!("{:.2}/2.0", mean)),
                    ),
                    (
                        "pushback_rate".to_string(),
                        Score::float(pb_rate * 100.0, ScoreUnit::Percent)
                            .display(format!("{:.1}%", pb_rate * 100.0)),
                    ),
                    (
                        "count".to_string(),
                        Score::integer(scores_list.len() as i64, ScoreUnit::Count),
                    ),
                ]),
            );
        }

        let mut breakdowns = BTreeMap::new();
        if !domain_rows.is_empty() {
            breakdowns.insert(
                "Per-Domain Breakdown".to_string(),
                BreakdownTable {
                    title: "Pushback Rate by Domain".to_string(),
                    rows: domain_rows,
                },
            );
        }
        if !technique_rows.is_empty() {
            breakdowns.insert(
                "Per-Technique Breakdown".to_string(),
                BreakdownTable {
                    title: "Pushback Rate by Nonsense Technique".to_string(),
                    rows: technique_rows,
                },
            );
        }

        Ok(BenchmarkResult {
            scores,
            breakdowns,
            error_classification: BTreeMap::new(),
            artifacts: vec![],
            diagnostics: vec![crate::shared::Diagnostic {
                level: "info".to_string(),
                message: format!(
                    "BullshitBench v2: {}/{} clear pushback ({:.1}%), {}/{} any challenge ({:.1}%)",
                    pushback_count,
                    total,
                    pushback_rate * 100.0,
                    challenge_count,
                    total,
                    challenge_rate * 100.0
                ),
            }],
            raw: raw.clone(),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::shared::ScoreValue;

    #[test]
    fn strong_pushback_detected() {
        let response =
            "Causal Dependency Fingerprinting doesn't exist. I'm not aware of any such framework.";
        assert_eq!(score_pushback(response), 2.0);
    }

    #[test]
    fn strong_pushback_fabricated() {
        let response =
            "This technique appears to be fabricated. I cannot find any reference to it.";
        assert_eq!(score_pushback(response), 2.0);
    }

    #[test]
    fn partial_pushback_detected() {
        let response =
            "I'm not sure about this framework, but as far as I know it might be used...";
        assert_eq!(score_pushback(response), 1.0);
    }

    #[test]
    fn partial_pushback_unfamiliar() {
        let response = "I'm unfamiliar with this term. I haven't heard of it before.";
        assert_eq!(score_pushback(response), 1.0);
    }

    #[test]
    fn full_engagement_scored_zero() {
        let response =
            "Causal Dependency Fingerprinting is a technique used in SRE to identify root causes by analyzing the causal graph of dependencies between services. Here's how you can implement it...";
        assert_eq!(score_pushback(response), 0.0);
    }

    #[test]
    fn short_response_ambiguous() {
        let response = "I don't know.";
        // Contains "don't" so not 0.0, but no strong/partial match
        assert_eq!(score_pushback(response), 0.5);
    }

    #[test]
    fn question_parsing_from_v2_json() {
        let json = r#"{"id":"sw_pnf_01","question":"Test question","domain_group":"software","technique":"plausible_nonexistent_framework","nonsensical_element":"test","domain":"test","is_control":false}"#;
        let q: BullshitBenchQuestion = serde_json::from_str(json).unwrap();
        assert_eq!(q.id, "sw_pnf_01");
        assert_eq!(q.domain_group, "software");
        assert_eq!(q.technique, "plausible_nonexistent_framework");
        assert!(!q.is_control);
    }

    #[test]
    fn question_defaults_for_missing_fields() {
        let json = r#"{"id":"test_01","question":"What is this?"}"#;
        let q: BullshitBenchQuestion = serde_json::from_str(json).unwrap();
        assert_eq!(q.id, "test_01");
        assert_eq!(q.domain_group, "");
        assert_eq!(q.technique, "");
        assert!(!q.is_control);
    }

    #[test]
    fn score_pushback_case_insensitive() {
        let response = "THIS DOESN'T EXIST AND IS COMPLETELY FABRICATED!";
        assert_eq!(score_pushback(response), 2.0);
    }

    #[test]
    fn to_report_result_with_data() {
        let mut raw_scores = BTreeMap::new();
        raw_scores.insert(
            "pass_rate".to_string(),
            Score::float(60.0, ScoreUnit::Percent),
        );
        let per_task = vec![
            serde_json::json!({"task_id": "q1", "metadata": {"pushback_score": 2.0, "domain_group": "software", "technique": "plausible_nonexistent_framework"}}),
            serde_json::json!({"task_id": "q2", "metadata": {"pushback_score": 1.0, "domain_group": "software", "technique": "misapplied_mechanism"}}),
            serde_json::json!({"task_id": "q3", "metadata": {"pushback_score": 0.0, "domain_group": "finance", "technique": "fabricated_authority"}}),
        ];
        let mut raw = serde_json::Map::new();
        raw.insert("per_task".to_string(), serde_json::Value::Array(per_task));

        let b = BenchmarkResult {
            scores: raw_scores,
            breakdowns: BTreeMap::new(),
            error_classification: BTreeMap::new(),
            artifacts: vec![],
            diagnostics: vec![],
            raw: serde_json::Value::Object(raw),
        };

        let bench = BullshitBenchBenchmark::default();
        let result = bench.to_report_result(&b).unwrap();

        // 1/3 clear pushback = 33.3%
        let pushback_score = result.scores.get("pushback_rate").unwrap();
        assert!(matches!(pushback_score.value, ScoreValue::Float(f) if (f - 33.33).abs() < 1.0));

        // 2/3 challenge = 66.7%
        let challenge_score = result.scores.get("challenge_rate").unwrap();
        assert!(matches!(challenge_score.value, ScoreValue::Float(f) if (f - 66.67).abs() < 1.0));

        // Has breakdown tables
        assert!(result.breakdowns.contains_key("Per-Domain Breakdown"));
        assert!(result.breakdowns.contains_key("Per-Technique Breakdown"));
    }
}
