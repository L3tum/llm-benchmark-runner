//! Linux Kernel Security Benchmark — Vulnerability-Introducing Commit Detection
//!
//! Binary classification: given a Linux kernel git commit (message + diff),
//! predict whether it introduced a bug/vulnerability that was later fixed.
//!
//! Dataset: pebblebed/kernel-vuln-dataset-full from HuggingFace
//! <https://huggingface.co/datasets/pebblebed/kernel-vuln-dataset-full>
//!
//! - 1.43M kernel commits with Fixes: tag mining labels
//! - Test split: 142,620 commits (8,000 positive, 134,620 negative; ~5.6% positive)
//! - Parquet format, downloaded at runtime and cached in ~/.cache/
//!
//! Metrics: Accuracy, Precision, Recall, F1 Score
//! Breakdowns: Per-subsystem, Per-class

use crate::benchmarks::Benchmark;
use crate::config::Model;
use crate::shared::{
    truncate, BenchmarkCategory, BenchmarkResult, BreakdownTable, Score, ScoreUnit, TaskResult,
};
use crate::token_tracker::TokenTracker;
use anyhow::{Context, Result};
use arrow::array::Array;
use arrow::record_batch::RecordBatch;
use rand::rngs::StdRng;
use rand::seq::SliceRandom;
use rand::SeedableRng;
use serde::Deserialize;
use std::collections::BTreeMap;
use std::fmt::Write;
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::Mutex;

// ---------------------------------------------------------------------------
// Config
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Deserialize)]
struct LksConfig {
    /// Total number of samples to evaluate (default: 200)
    #[serde(default = "default_num_samples")]
    num_samples: usize,
    /// Whether to include negative samples (default: true).
    /// If false, only positive (label=1) commits are used.
    #[serde(default = "default_true")]
    include_negative: bool,
    /// Max lines to include from the diff in the prompt (default: 200)
    #[serde(default = "default_diff_max_lines")]
    diff_max_lines: usize,
    /// Number of positive (label=1) samples to include (default: half of num_samples)
    #[serde(default)]
    num_positive: Option<usize>,
    /// Number of negative (label=0) samples to include (default: remainder)
    #[serde(default)]
    num_negative: Option<usize>,
    /// Random seed for reproducibility (default: 42)
    #[serde(default = "default_seed")]
    seed: u64,
}

fn default_num_samples() -> usize {
    200
}
fn default_true() -> bool {
    true
}
fn default_diff_max_lines() -> usize {
    200
}
fn default_seed() -> u64 {
    42
}

impl Default for LksConfig {
    fn default() -> Self {
        serde_json::from_value(serde_json::Value::Null).unwrap_or(Self {
            num_samples: default_num_samples(),
            include_negative: true,
            diff_max_lines: default_diff_max_lines(),
            num_positive: None,
            num_negative: None,
            seed: default_seed(),
        })
    }
}

// ---------------------------------------------------------------------------
// Data structures
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
struct KernelCommit {
    hash: String,
    subject: String,
    body: String,
    trailers: String,
    diff_raw: String,
    insertions: i64,
    deletions: i64,
    files_changed: i32,
    label: u8, // 1 = vuln-introducing, 0 = clean
    /// Derived from affected file path (e.g. "net/ipv4", "crypto", "fs")
    subsystem: String,
}

// ---------------------------------------------------------------------------
// State
// ---------------------------------------------------------------------------

pub struct LinuxKernelSecurityBenchmark {
    state: Mutex<LksState>,
}

struct LksState {
    items: Vec<KernelCommit>,
    current_idx: usize,
    config: LksConfig,
}

impl Default for LinuxKernelSecurityBenchmark {
    fn default() -> Self {
        Self {
            state: Mutex::new(LksState {
                items: Vec::new(),
                current_idx: 0,
                config: LksConfig::default(),
            }),
        }
    }
}

// ---------------------------------------------------------------------------
// Dataset download & Parquet reading
// ---------------------------------------------------------------------------

/// Download the test split parquet file from HuggingFace.
fn download_test_split(cache_dir: &Path) -> Result<PathBuf> {
    let parquet_path = cache_dir.join("test.parquet");
    if parquet_path.exists() {
        return Ok(parquet_path);
    }

    let url = "https://huggingface.co/datasets/pebblebed/kernel-vuln-dataset-full/resolve/main/default/test/test-00000-of-00001.parquet";
    println!("  Downloading kernel-vuln-dataset-full test split from HuggingFace...");

    let resp = reqwest::blocking::get(url)
        .context("failed to connect to HuggingFace")?
        .error_for_status()
        .context("HuggingFace returned an error")?;

    let bytes = resp.bytes().context("failed to read response body")?;

    fs::write(&parquet_path, &bytes).context("failed to write parquet file to cache")?;

    let size_mb = bytes.len() as f64 / 1_048_576.0;
    println!("  Downloaded test split ({:.1} MB)", size_mb);

    Ok(parquet_path)
}

/// Read the parquet file into a vector of KernelCommit structs.
fn read_parquet_commits(parquet_path: &Path) -> Result<Vec<KernelCommit>> {
    let file = fs::File::open(parquet_path).context("failed to open parquet file")?;
    use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
    let builder = ParquetRecordBatchReaderBuilder::try_new(file)
        .context("failed to create parquet batch reader builder")?;
    let reader = builder
        .build()
        .context("failed to build parquet batch reader")?;

    let mut commits = Vec::new();

    for batch_result in reader {
        let batch = batch_result?;
        let commits_in_batch = extract_commits_from_batch(&batch)?;
        commits.extend(commits_in_batch);
    }

    Ok(commits)
}

/// Extract KernelCommit structs from an Arrow RecordBatch.
fn extract_commits_from_batch(batch: &RecordBatch) -> Result<Vec<KernelCommit>> {
    let num_rows = batch.num_rows();
    let mut commits = Vec::with_capacity(num_rows);

    // Find column indices — return error if a required column is missing
    let col_index = |name: &str| -> Result<usize> {
        batch
            .schema()
            .column_with_name(name)
            .map(|(i, _)| i)
            .ok_or_else(|| {
                anyhow::anyhow!(
                    "Parquet schema missing expected column: '{}'. \
                 Available columns: {:?}",
                    name,
                    batch.schema().fields()
                )
            })
    };

    let hash_idx = col_index("hash")?;
    let subject_idx = col_index("subject")?;
    let body_idx = col_index("body")?;
    let trailers_idx = col_index("trailers")?;
    let diff_idx = col_index("diff_raw")?;
    let ins_idx = col_index("insertions")?;
    let del_idx = col_index("deletions")?;
    let files_idx = col_index("files_changed")?;
    let label_idx = col_index("label")?;

    for row in 0..num_rows {
        let hash = string_from_column(batch.column(hash_idx), row).unwrap_or_default();
        let subject = string_from_column(batch.column(subject_idx), row).unwrap_or_default();
        let body = string_from_column(batch.column(body_idx), row).unwrap_or_default();
        let trailers = string_from_column(batch.column(trailers_idx), row).unwrap_or_default();
        let diff_raw = string_from_column(batch.column(diff_idx), row).unwrap_or_default();
        let insertions = i64_from_column(batch.column(ins_idx), row).unwrap_or(0);
        let deletions = i64_from_column(batch.column(del_idx), row).unwrap_or(0);
        let files_changed = i32_from_column(batch.column(files_idx), row).unwrap_or(0);
        let label = u8_from_column(batch.column(label_idx), row).unwrap_or(0);

        // Skip merge commits (empty diff) with label=1
        if diff_raw.is_empty() && label == 1 {
            continue;
        }

        // Extract subsystem from diff (first changed file path)
        let subsystem = extract_subsystem_from_diff(&diff_raw);

        commits.push(KernelCommit {
            hash,
            subject,
            body,
            trailers,
            diff_raw,
            insertions,
            deletions,
            files_changed,
            label,
            subsystem,
        });
    }

    Ok(commits)
}

fn string_from_column(column: &dyn Array, index: usize) -> Option<String> {
    use arrow::array::StringArray;
    let arr = column.as_any().downcast_ref::<StringArray>()?;
    Some(arr.value(index).to_string())
}

fn i64_from_column(column: &dyn Array, index: usize) -> Option<i64> {
    use arrow::array::Int64Array;
    let arr = column.as_any().downcast_ref::<Int64Array>()?;
    Some(arr.value(index))
}

fn i32_from_column(column: &dyn Array, index: usize) -> Option<i32> {
    use arrow::array::Int32Array;
    let arr = column.as_any().downcast_ref::<Int32Array>()?;
    Some(arr.value(index))
}

fn u8_from_column(column: &dyn Array, index: usize) -> Option<u8> {
    use arrow::array::{Int16Array, Int32Array, Int64Array, Int8Array};
    if let Some(arr) = column.as_any().downcast_ref::<Int8Array>() {
        return Some(arr.value(index) as u8);
    }
    if let Some(arr) = column.as_any().downcast_ref::<Int16Array>() {
        return Some(arr.value(index) as u8);
    }
    if let Some(arr) = column.as_any().downcast_ref::<Int32Array>() {
        return Some(arr.value(index) as u8);
    }
    if let Some(arr) = column.as_any().downcast_ref::<Int64Array>() {
        return Some(arr.value(index) as u8);
    }
    None
}

/// Extract the subsystem from diff file paths (e.g., "net/ipv4", "crypto").
fn extract_subsystem_from_diff(diff: &str) -> String {
    for line in diff.lines() {
        if let Some(path) = line.strip_prefix("+++ b/") {
            let parts: Vec<&str> = path.split('/').collect();
            if parts.len() >= 3 {
                // Two or more directories (e.g., "net/ipv4/tcp.c" → "net/ipv4")
                return format!("{}/{}", parts[0], parts[1]);
            } else if !parts.is_empty() {
                // Single directory (e.g., "crypto/sha256.c" → "crypto")
                return parts[0].to_string();
            }
        }
    }
    "unknown".to_string()
}

/// Load and sample the dataset according to config.
fn load_and_sample_dataset(config: &LksConfig) -> Result<Vec<KernelCommit>> {
    let cache_dir = dirs::cache_dir()
        .unwrap_or_else(|| PathBuf::from("/tmp"))
        .join("llm-benchmark-runner")
        .join("kernel-vuln-dataset");

    fs::create_dir_all(&cache_dir).context("failed to create cache dir")?;

    let parquet_path = download_test_split(&cache_dir)?;
    let all_commits = read_parquet_commits(&parquet_path)?;

    println!("  Loaded {} commits from test split", all_commits.len());

    // Separate positive and negative samples
    let mut positive: Vec<KernelCommit> = all_commits
        .iter()
        .filter(|c| c.label == 1)
        .cloned()
        .collect();
    let mut negative: Vec<KernelCommit> = all_commits
        .iter()
        .filter(|c| c.label == 0)
        .cloned()
        .collect();

    println!(
        "  Positive (vuln): {}, Negative (clean): {}",
        positive.len(),
        negative.len()
    );

    // Shuffle with deterministic seed
    let mut rng = StdRng::seed_from_u64(config.seed);
    positive.shuffle(&mut rng);
    negative.shuffle(&mut rng);

    // Determine sample counts
    let num_pos = config.num_positive.unwrap_or(config.num_samples / 2);
    let num_neg = if config.include_negative {
        config
            .num_negative
            .unwrap_or(config.num_samples.saturating_sub(num_pos))
    } else {
        0
    };

    // Sample (capture lengths before consuming the iterators)
    let pos_len = positive.len();
    let neg_len = negative.len();
    let sampled_pos: Vec<KernelCommit> = positive.into_iter().take(num_pos.min(pos_len)).collect();
    let sampled_neg: Vec<KernelCommit> = negative.into_iter().take(num_neg.min(neg_len)).collect();

    let mut sampled: Vec<KernelCommit> = Vec::with_capacity(sampled_pos.len() + sampled_neg.len());
    sampled.extend(sampled_pos);
    sampled.extend(sampled_neg);

    // Final shuffle to interleave positive/negative
    sampled.shuffle(&mut rng);

    println!(
        "  Sampled: {} positive + {} negative = {} total",
        sampled.iter().filter(|c| c.label == 1).count(),
        sampled.iter().filter(|c| c.label == 0).count(),
        sampled.len()
    );

    Ok(sampled)
}

// ---------------------------------------------------------------------------
// Prompting & response parsing
// ---------------------------------------------------------------------------

const SYSTEM_PROMPT: &str =
    "You are an expert Linux kernel security researcher specializing in vulnerability-introducing \
     commit detection. Your task is to analyze git commits and determine whether they introduced \
     a bug or vulnerability that was later fixed.";

fn build_classification_prompt(commit: &KernelCommit, diff_max_lines: usize) -> String {
    let diff_display = truncate_diff(&commit.diff_raw, diff_max_lines);

    let trailers_display = if commit.trailers.is_empty() {
        String::new()
    } else {
        format!("\n\nTrailers:\n{}", commit.trailers)
    };

    let commit_message = if commit.body.is_empty() {
        commit.subject.clone()
    } else {
        format!("{}\n\n{}", commit.subject, commit.body)
    };

    format!(
        "Analyze the following Linux kernel git commit. Determine whether this commit \
         introduced a vulnerability or bug that was later fixed.

## Commit: {}
## Subject: {}
## Changes: +{} / -{} lines across {} file(s)
## Subsystem: {}

Commit Message:
{}{}

Diff:
```diff
{}
```

Answer with your prediction and brief reasoning:
PREDICTION: YES or NO
CONFIDENCE: HIGH / MEDIUM / LOW
REASONING: <brief explanation>",
        &commit.hash[..commit.hash.len().min(12)],
        commit.subject,
        commit.insertions,
        commit.deletions,
        commit.files_changed,
        commit.subsystem,
        commit_message,
        trailers_display,
        diff_display,
    )
}

fn truncate_diff(diff: &str, max_lines: usize) -> String {
    let lines: Vec<&str> = diff.lines().collect();
    if lines.len() <= max_lines {
        return diff.to_string();
    }
    let mut truncated = lines[..max_lines].join("\n");
    write!(
        truncated,
        "\n... ({} more lines truncated)",
        lines.len() - max_lines
    )
    .unwrap();
    truncated
}

fn parse_classification_response(response: &str) -> (bool, String) {
    let upper = response.to_uppercase();

    // Look for PREDICTION line
    let prediction = if upper.contains("PREDICTION: YES")
        || upper.contains("PREDICTION: Y ")
        || upper.contains("PREDICTION:Y")
    {
        true
    } else if upper.contains("PREDICTION: NO")
        || upper.contains("PREDICTION: N ")
        || upper.contains("PREDICTION:N")
    {
        false
    } else {
        // Fallback: look for common patterns
        let lower = response.to_lowercase();
        let yes_signals = [
            "introduced a vulnerability",
            "introduced a bug",
            "this commit is vulnerable",
            "yes, this introduced",
            "yes, this is",
            "this is a vulnerability",
            "this is a bug",
            "vulnerability-introducing",
            "bug-introducing",
            "this commit introduced",
        ];
        let no_signals = [
            "no vulnerability",
            "no bug",
            "clean commit",
            "this is clean",
            "does not introduce",
            "no defect",
            "no vulnerability introduced",
            "this commit does not",
            "appears to be a normal",
        ];

        let yes_score = yes_signals.iter().filter(|s| lower.contains(**s)).count();
        let no_score = no_signals.iter().filter(|s| lower.contains(**s)).count();
        yes_score > no_score
    };

    // Extract reasoning
    let reasoning = if let Some(start) = response.find("REASONING:") {
        response[start + "REASONING:".len()..].trim().to_string()
    } else {
        let lines: Vec<&str> = response.lines().collect();
        if lines.len() > 3 {
            lines[lines.len() - 3..].join(" ")
        } else {
            response.to_string()
        }
    };

    (prediction, reasoning)
}

// ---------------------------------------------------------------------------
// Metrics computation
// ---------------------------------------------------------------------------

fn compute_metrics(results: &[TaskResult]) -> (f64, f64, f64, f64, usize, usize, usize, usize) {
    let mut tp = 0;
    let mut tn = 0;
    let mut fp = 0;
    let mut fn_ = 0;

    for r in results {
        // Read prediction and ground truth from metadata — NOT from r.passed,
        // because r.passed means "was the answer correct", not "was it predicted positive".
        let meta = r.metadata.as_ref().expect("metadata required for metrics");
        let predicted_positive = meta
            .get("predicted_positive")
            .and_then(|p| p.as_bool())
            .unwrap_or(false);
        let actual_positive = meta
            .get("actual_label")
            .and_then(|l| l.as_u64())
            .unwrap_or(0)
            == 1;

        if predicted_positive && actual_positive {
            tp += 1;
        } else if !predicted_positive && !actual_positive {
            tn += 1;
        } else if predicted_positive && !actual_positive {
            fp += 1;
        } else {
            fn_ += 1;
        }
    }

    let total = tp + tn + fp + fn_;
    let accuracy = if total > 0 {
        (tp + tn) as f64 / total as f64
    } else {
        0.0
    };
    let precision = if tp + fp > 0 {
        tp as f64 / (tp + fp) as f64
    } else {
        0.0
    };
    let recall = if tp + fn_ > 0 {
        tp as f64 / (tp + fn_) as f64
    } else {
        0.0
    };
    let f1 = if precision + recall > 0.0 {
        2.0 * precision * recall / (precision + recall)
    } else {
        0.0
    };

    (accuracy, precision, recall, f1, tp, tn, fp, fn_)
}

// ---------------------------------------------------------------------------
// Benchmark trait
// ---------------------------------------------------------------------------

impl Benchmark for LinuxKernelSecurityBenchmark {
    fn name(&self) -> &str {
        "linux_kernel_security"
    }

    fn display_name(&self) -> &'static str {
        "Linux Kernel Security (VCC)"
    }

    fn category(&self) -> BenchmarkCategory {
        BenchmarkCategory::Security
    }

    fn pre_execute(&self, config: &yaml_serde::Value) -> Result<()> {
        let cfg: LksConfig = if let Some(raw) = config.get("linux_kernel_security") {
            serde_json::from_value(serde_json::to_value(raw).unwrap_or_default())
                .unwrap_or_default()
        } else {
            serde_json::from_value(serde_json::to_value(config).unwrap_or_default())
                .unwrap_or_default()
        };

        let items = load_and_sample_dataset(&cfg)?;

        let mut state = self.state.lock().expect(crate::shared::MUTEX_PANIC_MSG);
        state.items = items;
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
        let (commit, cfg) = {
            let mut state = self.state.lock().expect(crate::shared::MUTEX_PANIC_MSG);
            if state.current_idx >= state.items.len() {
                return Ok(None);
            }
            let idx = state.current_idx;
            let commit = state.items[idx].clone();
            state.current_idx += 1;
            let cfg = state.config.clone();
            (commit, cfg)
        };

        let prompt = build_classification_prompt(&commit, cfg.diff_max_lines);
        let response = tracker.chat_completion(&model.model_name, SYSTEM_PROMPT, &prompt)?;

        let (predicted, reasoning) = parse_classification_response(&response);

        let correct = predicted == (commit.label == 1);
        let score = if correct { 1.0 } else { 0.0 };

        Ok(Some(
            TaskResult::new(
                commit.hash.clone(),
                correct,
                score,
                vec![
                    commit.subsystem.clone(),
                    if commit.label == 1 {
                        "positive"
                    } else {
                        "negative"
                    }
                    .to_string(),
                ],
            )
            .with_metadata(Some(serde_json::json!({
                "commit_hash": commit.hash,
                "actual_label": commit.label,
                "predicted_positive": predicted,
                "reasoning": truncate(&reasoning, 300),
                "files_changed": commit.files_changed,
                "insertions": commit.insertions,
                "deletions": commit.deletions,
                "subsystem": commit.subsystem,
            }))),
        ))
    }

    fn to_report_result(&self, b: &BenchmarkResult) -> Result<BenchmarkResult> {
        let per_task: Vec<&serde_json::Value> = b
            .raw
            .get("per_task")
            .and_then(|v| v.as_array())
            .map(|arr| arr.iter().collect())
            .unwrap_or_default();

        // Reconstruct TaskResults with metadata preserved for compute_metrics
        let results: Vec<TaskResult> = per_task
            .iter()
            .map(|t| {
                let meta = t.get("metadata").unwrap_or(&serde_json::Value::Null);
                let actual = meta
                    .get("actual_label")
                    .and_then(|l| l.as_u64())
                    .unwrap_or(0)
                    == 1;
                let predicted = meta
                    .get("predicted_positive")
                    .and_then(|p| p.as_bool())
                    .unwrap_or(false);
                let correct = predicted == actual;
                TaskResult::new(
                    meta.get("commit_hash")
                        .and_then(|h| h.as_str())
                        .unwrap_or("")
                        .to_string(),
                    correct,
                    if correct { 1.0 } else { 0.0 },
                    vec![],
                )
                .with_metadata(Some(meta.clone()))
            })
            .collect();

        let (accuracy, precision, recall, f1, tp, tn, fp, fn_) = compute_metrics(&results);
        let total = results.len();

        let mut scores = BTreeMap::new();
        scores.insert(
            "accuracy".to_string(),
            Score::float(accuracy * 100.0, ScoreUnit::Percent)
                .primary(true)
                .higher_is_better(true)
                .display(format!("{:.1}%", accuracy * 100.0)),
        );
        scores.insert(
            "precision".to_string(),
            Score::float(precision * 100.0, ScoreUnit::Percent)
                .higher_is_better(true)
                .display(format!("{:.1}%", precision * 100.0)),
        );
        scores.insert(
            "recall".to_string(),
            Score::float(recall * 100.0, ScoreUnit::Percent)
                .higher_is_better(true)
                .display(format!("{:.1}%", recall * 100.0)),
        );
        scores.insert(
            "f1_score".to_string(),
            Score::float(f1 * 100.0, ScoreUnit::Percent)
                .higher_is_better(true)
                .display(format!("{:.1}%", f1 * 100.0)),
        );
        scores.insert(
            "total_samples".to_string(),
            Score::integer(total as i64, ScoreUnit::Count),
        );

        let diagnostics = vec![crate::shared::Diagnostic {
            level: "info".to_string(),
            message: format!(
                "Confusion Matrix: TP={}, FP={}, FN={}, TN={}",
                tp, fp, fn_, tn
            ),
        }];

        // Per-subsystem breakdown
        let mut subsystem_data: BTreeMap<String, (usize, usize)> = BTreeMap::new();
        for task in &per_task {
            let meta = task.get("metadata").unwrap_or(&serde_json::Value::Null);
            let subsystem = meta
                .get("subsystem")
                .and_then(|s| s.as_str())
                .unwrap_or("unknown");
            let actual = meta
                .get("actual_label")
                .and_then(|l| l.as_u64())
                .unwrap_or(0)
                == 1;
            let predicted = meta
                .get("predicted_positive")
                .and_then(|p| p.as_bool())
                .unwrap_or(false);
            let correct = predicted == actual;
            let entry = subsystem_data
                .entry(subsystem.to_string())
                .or_insert((0, 0));
            entry.1 += 1;
            if correct {
                entry.0 += 1;
            }
        }

        let mut breakdowns = BTreeMap::new();
        if !subsystem_data.is_empty() {
            let mut rows = BTreeMap::new();
            for (subsystem, (correct, total)) in &subsystem_data {
                let acc = *correct as f64 / *total as f64;
                rows.insert(
                    subsystem.clone(),
                    BTreeMap::from_iter([
                        (
                            "accuracy".to_string(),
                            Score::float(acc * 100.0, ScoreUnit::Percent)
                                .display(format!("{:.1}%", acc * 100.0)),
                        ),
                        (
                            "correct".to_string(),
                            Score::integer(*correct as i64, ScoreUnit::Count),
                        ),
                        (
                            "total".to_string(),
                            Score::integer(*total as i64, ScoreUnit::Count),
                        ),
                    ]),
                );
            }
            breakdowns.insert(
                "Per-Subsystem Breakdown".to_string(),
                BreakdownTable {
                    title: "Accuracy by Kernel Subsystem".to_string(),
                    rows,
                },
            );
        }

        // Per-class breakdown
        let mut label_data: BTreeMap<String, (usize, usize)> = BTreeMap::new();
        for task in &per_task {
            let meta = task.get("metadata").unwrap_or(&serde_json::Value::Null);
            let label = meta
                .get("actual_label")
                .and_then(|l| l.as_u64())
                .unwrap_or(0);
            let label_str = if label == 1 {
                "positive (vuln)"
            } else {
                "negative (clean)"
            };
            let actual = label == 1;
            let predicted = meta
                .get("predicted_positive")
                .and_then(|p| p.as_bool())
                .unwrap_or(false);
            let correct = predicted == actual;
            let entry = label_data.entry(label_str.to_string()).or_insert((0, 0));
            entry.1 += 1;
            if correct {
                entry.0 += 1;
            }
        }

        if !label_data.is_empty() {
            let mut rows = BTreeMap::new();
            for (label, (correct, total)) in &label_data {
                let acc = *correct as f64 / *total as f64;
                rows.insert(
                    label.clone(),
                    BTreeMap::from_iter([
                        (
                            "accuracy".to_string(),
                            Score::float(acc * 100.0, ScoreUnit::Percent)
                                .display(format!("{:.1}%", acc * 100.0)),
                        ),
                        (
                            "correct".to_string(),
                            Score::integer(*correct as i64, ScoreUnit::Count),
                        ),
                        (
                            "total".to_string(),
                            Score::integer(*total as i64, ScoreUnit::Count),
                        ),
                    ]),
                );
            }
            breakdowns.insert(
                "Per-Class Breakdown".to_string(),
                BreakdownTable {
                    title: "Accuracy by Label Class".to_string(),
                    rows,
                },
            );
        }

        Ok(BenchmarkResult {
            scores,
            breakdowns,
            error_classification: BTreeMap::new(),
            artifacts: vec![],
            diagnostics,
            raw: b.raw.clone(),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_classification_response_yes() {
        let response = "This commit looks vulnerable.\nPREDICTION: YES\nCONFIDENCE: HIGH\nREASONING: Missing bounds check";
        let (predicted, reasoning) = parse_classification_response(response);
        assert!(predicted);
        assert!(reasoning.contains("bounds check"));
    }

    #[test]
    fn parse_classification_response_no() {
        let response = "This is a clean commit with no issues.\nPREDICTION: NO\nCONFIDENCE: HIGH\nREASONING: Normal feature addition";
        let (predicted, reasoning) = parse_classification_response(response);
        assert!(!predicted);
        assert!(reasoning.contains("feature"));
    }

    #[test]
    fn parse_classification_response_fallback_yes() {
        let response = "This commit introduced a vulnerability in the kernel.";
        let (predicted, _) = parse_classification_response(response);
        assert!(predicted);
    }

    #[test]
    fn parse_classification_response_fallback_no() {
        let response = "No vulnerability was introduced. This is a clean commit.";
        let (predicted, _) = parse_classification_response(response);
        assert!(!predicted);
    }

    #[test]
    fn compute_metrics_perfect() {
        let results = vec![
            TaskResult::new("h1", true, 1.0, vec![]).with_metadata(Some(
                serde_json::json!({"actual_label": 1, "predicted_positive": true}),
            )),
            TaskResult::new("h2", true, 1.0, vec![]).with_metadata(Some(
                serde_json::json!({"actual_label": 1, "predicted_positive": true}),
            )),
            TaskResult::new("h3", true, 1.0, vec![]).with_metadata(Some(
                serde_json::json!({"actual_label": 1, "predicted_positive": true}),
            )),
            TaskResult::new("h4", true, 1.0, vec![]).with_metadata(Some(
                serde_json::json!({"actual_label": 1, "predicted_positive": true}),
            )),
            TaskResult::new("h5", true, 1.0, vec![]).with_metadata(Some(
                serde_json::json!({"actual_label": 1, "predicted_positive": true}),
            )),
            TaskResult::new("h6", true, 1.0, vec![]).with_metadata(Some(
                serde_json::json!({"actual_label": 0, "predicted_positive": false}),
            )),
            TaskResult::new("h7", true, 1.0, vec![]).with_metadata(Some(
                serde_json::json!({"actual_label": 0, "predicted_positive": false}),
            )),
            TaskResult::new("h8", true, 1.0, vec![]).with_metadata(Some(
                serde_json::json!({"actual_label": 0, "predicted_positive": false}),
            )),
            TaskResult::new("h9", true, 1.0, vec![]).with_metadata(Some(
                serde_json::json!({"actual_label": 0, "predicted_positive": false}),
            )),
            TaskResult::new("h10", true, 1.0, vec![]).with_metadata(Some(
                serde_json::json!({"actual_label": 0, "predicted_positive": false}),
            )),
        ];
        let (acc, prec, rec, f1, tp, tn, fp, fn_) = compute_metrics(&results);
        assert_eq!(tp, 5);
        assert_eq!(tn, 5);
        assert_eq!(fp, 0);
        assert_eq!(fn_, 0);
        assert!((acc - 1.0).abs() < 0.01);
        assert!((prec - 1.0).abs() < 0.01);
        assert!((rec - 1.0).abs() < 0.01);
        assert!((f1 - 1.0).abs() < 0.01);
    }

    #[test]
    fn compute_metrics_with_errors() {
        // 3 TP, 2 FP, 2 FN, 3 TN => acc=0.6, prec=0.6, rec=0.6
        let results = vec![
            // 3 TP: actual=1, predicted=true
            TaskResult::new("h1", true, 1.0, vec![]).with_metadata(Some(
                serde_json::json!({"actual_label": 1, "predicted_positive": true}),
            )),
            TaskResult::new("h2", true, 1.0, vec![]).with_metadata(Some(
                serde_json::json!({"actual_label": 1, "predicted_positive": true}),
            )),
            TaskResult::new("h3", true, 1.0, vec![]).with_metadata(Some(
                serde_json::json!({"actual_label": 1, "predicted_positive": true}),
            )),
            // 2 FN: actual=1, predicted=false
            TaskResult::new("h4", false, 0.0, vec![]).with_metadata(Some(
                serde_json::json!({"actual_label": 1, "predicted_positive": false}),
            )),
            TaskResult::new("h5", false, 0.0, vec![]).with_metadata(Some(
                serde_json::json!({"actual_label": 1, "predicted_positive": false}),
            )),
            // 2 FP: actual=0, predicted=true
            TaskResult::new("h6", false, 0.0, vec![]).with_metadata(Some(
                serde_json::json!({"actual_label": 0, "predicted_positive": true}),
            )),
            TaskResult::new("h7", false, 0.0, vec![]).with_metadata(Some(
                serde_json::json!({"actual_label": 0, "predicted_positive": true}),
            )),
            // 3 TN: actual=0, predicted=false
            TaskResult::new("h8", true, 1.0, vec![]).with_metadata(Some(
                serde_json::json!({"actual_label": 0, "predicted_positive": false}),
            )),
            TaskResult::new("h9", true, 1.0, vec![]).with_metadata(Some(
                serde_json::json!({"actual_label": 0, "predicted_positive": false}),
            )),
            TaskResult::new("h10", true, 1.0, vec![]).with_metadata(Some(
                serde_json::json!({"actual_label": 0, "predicted_positive": false}),
            )),
        ];
        let (acc, prec, rec, _f1, tp, tn, fp, fn_) = compute_metrics(&results);
        assert_eq!(tp, 3);
        assert_eq!(fp, 2);
        assert_eq!(fn_, 2);
        assert_eq!(tn, 3);
        assert!((acc - 0.6).abs() < 0.01);
        assert!((prec - 0.6).abs() < 0.01);
        assert!((rec - 0.6).abs() < 0.01);
    }

    #[test]
    fn test_extract_subsystem_from_diff() {
        let diff = "--- a/net/ipv4/tcp.c\n+++ b/net/ipv4/tcp.c\n@@ -1,2 +1,2 @@";
        assert_eq!(extract_subsystem_from_diff(diff), "net/ipv4");

        let diff2 = "--- a/crypto/sha256.c\n+++ b/crypto/sha256.c";
        assert_eq!(extract_subsystem_from_diff(diff2), "crypto");

        let diff3 = "--- a/init/main.c\n+++ b/init/main.c";
        assert_eq!(extract_subsystem_from_diff(diff3), "init");
    }

    #[test]
    fn test_truncate_diff_respects_limit() {
        let diff = (0..100)
            .map(|i| format!("line {}", i))
            .collect::<Vec<_>>()
            .join("\n");
        let truncated = truncate_diff(&diff, 10);
        assert!(truncated.contains("... (90 more lines truncated)"));
    }

    #[test]
    fn config_defaults() {
        let cfg: LksConfig = LksConfig::default();
        assert_eq!(cfg.num_samples, 200);
        assert!(cfg.include_negative);
        assert_eq!(cfg.diff_max_lines, 200);
        assert_eq!(cfg.seed, 42);
    }
}
