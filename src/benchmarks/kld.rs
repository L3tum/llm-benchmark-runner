use crate::benchmarks::mmlu_pro::MmluProBenchmark;
use crate::benchmarks::Benchmark;
use crate::client::LogprobEntry;
use crate::config::Model;
use crate::shared::{
    BenchmarkCategory, BenchmarkResult, BreakdownTable, Diagnostic, Score, ScoreUnit, TaskResult,
    TestAggregate,
};
use crate::token_tracker::TokenTracker;
use anyhow::Result;
use std::collections::{BTreeMap, HashMap};
use std::fs;
use std::sync::Mutex;

fn load_prompts_from_file(path: &str, num_prompts: usize) -> Result<Vec<String>> {
    let content = fs::read_to_string(path)?;
    // Try JSON array first
    if let Ok(strings) = serde_json::from_str::<Vec<String>>(&content) {
        Ok(strings.into_iter().take(num_prompts).collect())
    } else {
        // Fallback: newline-separated text, skip empty lines
        let prompts: Vec<String> = content
            .lines()
            .filter(|line| !line.trim().is_empty())
            .take(num_prompts)
            .map(|s| s.to_string())
            .collect();
        Ok(prompts)
    }
}

pub struct KldBenchmark {
    state: Mutex<KldState>,
}

struct KldState {
    prompts: Vec<String>,
    current_idx: usize,
}

impl Default for KldBenchmark {
    fn default() -> Self {
        Self {
            state: Mutex::new(KldState {
                prompts: Vec::new(),
                current_idx: 0,
            }),
        }
    }
}

fn compute_kl_from_logprobs(logprobs_a: &[LogprobEntry], logprobs_b: &[LogprobEntry]) -> f64 {
    if logprobs_a.is_empty() || logprobs_b.is_empty() {
        return f64::INFINITY;
    }

    // Convert to probability distributions over common tokens
    fn logprobs_to_dist(lp: &[LogprobEntry]) -> HashMap<&str, f64> {
        let mut dist: HashMap<&str, f64> = HashMap::new();
        let max_logprob = lp
            .iter()
            .map(|e| e.logprob)
            .fold(f64::NEG_INFINITY, f64::max);
        for entry in lp {
            dist.insert(&entry.token, (entry.logprob - max_logprob).exp());
        }
        let total: f64 = dist.values().sum();
        if total == 0.0 {
            return HashMap::new();
        }
        for val in dist.values_mut() {
            *val /= total;
        }
        dist
    }

    let dist_a = logprobs_to_dist(logprobs_a);
    let dist_b = logprobs_to_dist(logprobs_b);

    let mut kl = 0.0;
    for (&token, p) in &dist_a {
        let q = *dist_b.get(token).unwrap_or(&0.0);
        let p = *p;
        if p > 0.0 {
            if q > 0.0 {
                kl += p * (p / q).ln();
            } else {
                return f64::INFINITY;
            }
        }
    }
    kl
}

impl Benchmark for KldBenchmark {
    fn name(&self) -> &str {
        "kld"
    }

    fn display_name(&self) -> &'static str {
        "KLD"
    }

    fn category(&self) -> BenchmarkCategory {
        BenchmarkCategory::Similarity
    }

    fn to_report_result(&self, b: &BenchmarkResult) -> Result<BenchmarkResult> {
        let raw = &b.raw;

        let num_prompts = raw.get("num_prompts").and_then(|v| v.as_i64()).unwrap_or(0);
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
            "num_prompts_evaluated".to_string(),
            Score::integer(num_prompts, ScoreUnit::Count).primary(true),
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

        let diagnostics = vec![Diagnostic {
            level: "info".to_string(),
            message: "KLD score (similarity to other models) is shown in the aggregate/pairwise \
                      table. This per-model result only shows execution statistics."
                .to_string(),
        }];

        Ok(BenchmarkResult {
            scores,
            breakdowns: BTreeMap::new(),
            error_classification: BTreeMap::new(),
            artifacts: vec![],
            diagnostics,
            raw: raw.clone(),
        })
    }

    fn to_report_aggregate(&self, b: &BenchmarkResult) -> Result<Option<TestAggregate>> {
        let raw = &b.raw;

        // Extract pairwise KLD scores
        if let Some(pairwise) = raw.get("pairwise").and_then(|v| v.as_object()) {
            let mut rows = BTreeMap::new();
            for (pair_key, pair_data) in pairwise {
                if let Some(avg_kld) = pair_data.get("avg_kld").and_then(|v| v.as_f64()) {
                    let num_prompts = pair_data
                        .get("num_prompts_evaluated")
                        .and_then(|v| v.as_i64())
                        .unwrap_or(0);
                    let mut row_scores = BTreeMap::new();
                    row_scores.insert("avg_kld".to_string(), Score::float(avg_kld, ScoreUnit::Kld));
                    row_scores.insert(
                        "num_prompts_evaluated".to_string(),
                        Score::integer(num_prompts, ScoreUnit::Count),
                    );
                    rows.insert(pair_key.clone(), row_scores);
                }
            }
            if !rows.is_empty() {
                return Ok(Some(TestAggregate {
                    scores: BTreeMap::new(),
                    breakdowns: BTreeMap::from([(
                        "pairwise_kld".to_string(),
                        BreakdownTable {
                            title: "Pairwise KLD".to_string(),
                            rows,
                        },
                    )]),
                    raw: raw.clone(),
                }));
            }
        }

        Ok(None)
    }

    fn pre_execute(&self, config: &yaml_serde::Value) -> Result<()> {
        let num_prompts: usize = config
            .get("num_prompts")
            .and_then(|v| v.as_i64())
            .unwrap_or(10) as usize;
        let prompt_source = config
            .get("prompt_source")
            .and_then(|v| v.as_str())
            .unwrap_or("mmlu");
        let custom_prompts_path = config.get("custom_prompts_path").and_then(|v| v.as_str());

        let prompts: Vec<String> = match custom_prompts_path {
            Some(path) => {
                println!("  Loading custom prompts from {}", path);
                load_prompts_from_file(path, num_prompts)?
            }
            None if prompt_source == "mmlu" => {
                println!("  Using MMLU-Pro test prompts for KLD");
                let mmlu = MmluProBenchmark::default();
                mmlu.pre_execute(&yaml_serde::Value::Null)?;
                let test_path = mmlu.download_dataset("test")?;
                let items = mmlu.load_dataset(&test_path)?;
                items
                    .into_iter()
                    .take(num_prompts)
                    .map(|i| i.question)
                    .collect()
            }
            None => {
                return Err(anyhow::anyhow!(
                    "prompt_source '{}' not implemented; use 'mmlu' or set custom_prompts_path",
                    prompt_source
                ))
            }
        };

        if prompts.is_empty() {
            return Err(anyhow::anyhow!("No prompts loaded for KLD"));
        }

        let mut state = self.state.lock().expect(crate::shared::MUTEX_PANIC_MSG);
        state.prompts = prompts;
        state.current_idx = 0;
        Ok(())
    }

    fn execute_one(
        &self,
        model: &Model,
        _config: &yaml_serde::Value,
        tracker: &mut TokenTracker,
    ) -> Result<Option<TaskResult>> {
        let (prompt, idx) = {
            let mut state = self.state.lock().expect(crate::shared::MUTEX_PANIC_MSG);
            if state.current_idx >= state.prompts.len() {
                return Ok(None);
            }
            let idx = state.current_idx;
            let prompt = state.prompts[idx].clone();
            state.current_idx += 1;
            (prompt, idx)
        };

        match tracker.chat_completion_logprobs_with_usage(
            &model.model_name,
            "You are a helpful assistant.",
            &prompt,
        ) {
            Ok(logprobs) => {
                // Serialize logprobs to JSON for storage in metadata
                let logprobs_json: Vec<serde_json::Value> = logprobs
                    .iter()
                    .map(|e| serde_json::json!({ "token": e.token, "logprob": e.logprob }))
                    .collect();

                Ok(Some(
                    TaskResult::new(format!("task-{}", idx), true, 1.0, vec![]).with_metadata(
                        Some(serde_json::json!({
                            "prompt": prompt,
                            "logprobs": logprobs_json,
                        })),
                    ),
                ))
            }
            Err(e) => {
                eprintln!("  Error getting logprobs for {}: {}", model.display_name, e);
                Ok(Some(
                    TaskResult::new(format!("task-{}", idx), false, 0.0, vec![]).with_metadata(
                        Some(serde_json::json!({
                            "prompt": prompt,
                            "error": e.to_string(),
                            "logprobs": [],
                        })),
                    ),
                ))
            }
        }
    }

    fn post_execute(
        &self,
        model_results: &HashMap<String, BenchmarkResult>,
    ) -> Result<BenchmarkResult> {
        let mut all_logits: HashMap<String, Vec<Vec<LogprobEntry>>> = HashMap::new();
        let mut missing_models = Vec::new();

        for (name, result) in model_results {
            // Try per_task format first (new format)
            let entries =
                if let Some(per_task) = result.raw.get("per_task").and_then(|v| v.as_array()) {
                    let mut entries: Vec<Vec<LogprobEntry>> = Vec::new();
                    for task in per_task {
                        let mut logprobs: Vec<LogprobEntry> = Vec::new();
                        if let Some(lp_arr) = task.get("logprobs").and_then(|v| v.as_array()) {
                            for item in lp_arr {
                                if let (Some(token), Some(logprob)) =
                                    (item.get("token"), item.get("logprob"))
                                {
                                    if let (Some(token), Some(logprob)) =
                                        (token.as_str(), logprob.as_f64())
                                    {
                                        logprobs.push(LogprobEntry {
                                            token: token.to_string(),
                                            logprob,
                                        });
                                    }
                                }
                            }
                        }
                        entries.push(logprobs);
                    }
                    entries
                // Fallback: old kld array format
                } else if let Some(kld_arr) = result.raw.get("kld").and_then(|v| v.as_array()) {
                    let mut entries: Vec<Vec<LogprobEntry>> = Vec::new();
                    for arr in kld_arr {
                        if let Some(inner) = arr.as_array() {
                            let mut logprobs: Vec<LogprobEntry> = Vec::new();
                            for item in inner {
                                if let (Some(token), Some(logprob)) =
                                    (item.get("token"), item.get("logprob"))
                                {
                                    if let (Some(token), Some(logprob)) =
                                        (token.as_str(), logprob.as_f64())
                                    {
                                        logprobs.push(LogprobEntry {
                                            token: token.to_string(),
                                            logprob,
                                        });
                                    }
                                }
                            }
                            entries.push(logprobs);
                        }
                    }
                    entries
                } else {
                    missing_models.push(name.clone());
                    continue;
                };

            all_logits.insert(name.clone(), entries);
        }

        // If any model is missing KLD data, warn and continue so pairwise
        // comparisons among the available models are still computed.
        if !missing_models.is_empty() {
            eprintln!(
                "WARNING: KLD post-execute: missing KLD data for models: {} (these models failed to produce KLD scores, will be excluded from pairwise analysis)",
                missing_models.join(", ")
            );
        }

        let names: Vec<String> = all_logits.keys().cloned().collect();
        let mut pairwise = serde_json::Map::new();
        let mut kld_pairs: HashMap<(&str, &str), Vec<f64>> = HashMap::new();

        for (i, a_name) in names.iter().enumerate() {
            for b_name in names[i + 1..].iter() {
                let logits_a = &all_logits[a_name];
                let logits_b = &all_logits[b_name];
                let len = std::cmp::min(logits_a.len(), logits_b.len());
                let mut kld_values: Vec<f64> = Vec::new();
                for (logprobs_a, logprobs_b) in logits_a.iter().zip(logits_b.iter()).take(len) {
                    let kl = compute_kl_from_logprobs(logprobs_a, logprobs_b);
                    if kl.is_finite() {
                        kld_values.push(kl);
                    }
                }
                if !kld_values.is_empty() {
                    let avg_kld: f64 = kld_values.iter().sum::<f64>() / kld_values.len() as f64;
                    let key = format!("{}_vs_{}", a_name, b_name);
                    pairwise.insert(
                        key.clone(),
                        serde_json::json!({
                            "models": [a_name, b_name],
                            "avg_kld": avg_kld,
                            "num_prompts_evaluated": kld_values.len(),
                            "kld_values": kld_values,
                        }),
                    );
                    kld_pairs.insert((a_name.as_str(), b_name.as_str()), kld_values.clone());
                    kld_pairs.insert((b_name.as_str(), a_name.as_str()), kld_values);
                }
            }
        }

        // Compute average KLD to all other models
        let mut avg_kld_to_others = serde_json::Map::new();
        for model_name in &names {
            let mut kld_to_others = Vec::new();
            for other in &names {
                if other == model_name {
                    continue;
                }
                if let Some(values) = kld_pairs.get(&(model_name.as_str(), other.as_str())) {
                    let avg = values.iter().sum::<f64>() / values.len() as f64;
                    kld_to_others.push(avg);
                }
            }
            if !kld_to_others.is_empty() {
                let overall_avg = kld_to_others.iter().sum::<f64>() / kld_to_others.len() as f64;
                avg_kld_to_others.insert(
                    model_name.clone(),
                    serde_json::json!({
                        "avg_kld_to_others": overall_avg,
                        "klds": kld_to_others,
                    }),
                );
            }
        }
        pairwise.insert(
            "avg_kld_to_others".to_string(),
            serde_json::Value::Object(avg_kld_to_others),
        );

        Ok(BenchmarkResult {
            scores: BTreeMap::new(),
            breakdowns: BTreeMap::from([(
                "pairwise_kld".to_string(),
                BreakdownTable {
                    title: "Pairwise KLD".to_string(),
                    rows: pairwise
                        .iter()
                        .filter(|(k, _)| *k != "avg_kld_to_others")
                        .filter_map(|(key, data)| {
                            data.as_object().map(|obj| {
                                let avg_kld =
                                    obj.get("avg_kld").and_then(|v| v.as_f64()).unwrap_or(0.0);
                                let num_prompts = obj
                                    .get("num_prompts_evaluated")
                                    .and_then(|v| v.as_i64())
                                    .unwrap_or(0);
                                let row_scores = BTreeMap::from([
                                    ("avg_kld".to_string(), Score::float(avg_kld, ScoreUnit::Kld)),
                                    (
                                        "num_prompts_evaluated".to_string(),
                                        Score::integer(num_prompts, ScoreUnit::Count),
                                    ),
                                ]);
                                (key.clone(), row_scores)
                            })
                        })
                        .collect(),
                },
            )]),
            error_classification: BTreeMap::new(),
            artifacts: vec![],
            diagnostics: vec![],
            raw: serde_json::json!(pairwise),
        })
    }
}
