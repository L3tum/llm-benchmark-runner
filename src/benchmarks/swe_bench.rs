use crate::benchmarks::Benchmark;
use crate::config::Model;
use crate::docker_runner::{DockerBuildConfig, DockerMount, DockerRunConfig, DockerRunner};
use crate::shared::{
    truncate, BenchmarkCategory, BenchmarkResult, BreakdownTable, Score, ScoreUnit, TaskResult,
};
use crate::token_tracker::TokenTracker;
use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use serde_json::Value as JsonValue;
use std::collections::{BTreeMap, HashMap, HashSet};
use std::fs;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::Mutex;

/// Maximum size of a generated patch in bytes (1 MB).
const MAX_PATCH_BYTES: usize = 1_048_576;

/// Maximum allowed iterations for bash agent mode.
const MAX_ALLOWED_ITERATIONS: usize = 200;

// Agent commands run inside a Docker sandbox with cap_drop=ALL, network_none,
// read_only_root, PID limits, and memory limits — no allowlist needed.

#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "kebab-case")]
enum AgentMode {
    #[default]
    ZeroShot,
    Bash,
}

impl std::str::FromStr for AgentMode {
    type Err = anyhow::Error;
    fn from_str(s: &str) -> Result<Self> {
        match s {
            "zero-shot" | "zero_shot" | "zero" => Ok(Self::ZeroShot),
            "bash" | "agent" | "loop" => Ok(Self::Bash),
            other => Err(anyhow::anyhow!(
                "Unknown agent_mode: '{}'. Use 'zero-shot' or 'bash'.",
                other
            )),
        }
    }
}

pub struct SweBenchBenchmark {
    state: Mutex<SweBenchState>,
}

struct SweBenchState {
    items: Vec<SweBenchInstance>,
    current_idx: usize,
    /// Generated patches per instance, stored during execute_one for batch evaluation
    generated_patches: HashMap<usize, String>,
    /// Dataset variant for this benchmark (set during pre_execute)
    dataset: SweBenchDataset,
    /// Config for agent mode support
    config: Option<SweBenchConfig>,
}

impl Default for SweBenchBenchmark {
    fn default() -> Self {
        Self {
            state: Mutex::new(SweBenchState {
                items: Vec::new(),
                current_idx: 0,
                generated_patches: HashMap::new(),
                dataset: SweBenchDataset::Basic,
                config: None,
            }),
        }
    }
}
pub struct SweBenchVerifiedBenchmark {
    state: Mutex<SweBenchVerifiedState>,
}

struct SweBenchVerifiedState {
    items: Vec<SweBenchInstance>,
    current_idx: usize,
    generated_patches: HashMap<usize, String>,
    config: Option<SweBenchConfig>,
}

impl Default for SweBenchVerifiedBenchmark {
    fn default() -> Self {
        Self {
            state: Mutex::new(SweBenchVerifiedState {
                items: Vec::new(),
                current_idx: 0,
                generated_patches: HashMap::new(),
                config: None,
            }),
        }
    }
}
pub struct SweBenchProBenchmark {
    state: Mutex<SweBenchProState>,
}

struct SweBenchProState {
    items: Vec<SweBenchInstance>,
    current_idx: usize,
    generated_patches: HashMap<usize, String>,
    config: Option<SweBenchConfig>,
}

impl Default for SweBenchProBenchmark {
    fn default() -> Self {
        Self {
            state: Mutex::new(SweBenchProState {
                items: Vec::new(),
                current_idx: 0,
                generated_patches: HashMap::new(),
                config: None,
            }),
        }
    }
}

/// SWE-bench Multilingual: 300 tasks across 9 programming languages and 42 repositories.
/// Uses the same evaluation harness as SWE-Bench, but covers C, C++, Go, Java,
/// JavaScript/TypeScript, PHP, Ruby, and Rust.
pub struct SweBenchMultilingualBenchmark {
    state: Mutex<SweBenchMultilingualState>,
}

struct SweBenchMultilingualState {
    items: Vec<SweBenchInstance>,
    current_idx: usize,
    generated_patches: HashMap<usize, String>,
    config: Option<SweBenchConfig>,
}

impl Default for SweBenchMultilingualBenchmark {
    fn default() -> Self {
        Self {
            state: Mutex::new(SweBenchMultilingualState {
                items: Vec::new(),
                current_idx: 0,
                generated_patches: HashMap::new(),
                config: None,
            }),
        }
    }
}

#[derive(Debug, Clone, Copy)]
enum SweBenchDataset {
    Basic,
    Verified,
    Pro,
    Multilingual,
}

impl SweBenchDataset {
    #[allow(dead_code)] // used for registry naming; currently resolved from enum variant
    fn benchmark_name(self) -> &'static str {
        match self {
            Self::Basic => "swebench",
            Self::Verified => "swebench_verified",
            Self::Pro => "swebench_pro",
            Self::Multilingual => "swebench_multilingual",
        }
    }

    fn default_dataset_id(self) -> &'static str {
        match self {
            Self::Basic => "princeton-nlp/SWE-bench",
            Self::Verified => "princeton-nlp/SWE-bench_Verified",
            // Pro access/naming can be gated; users can override with benchmark.swebench_pro.dataset_id.
            Self::Pro => "SWE-bench/SWE-bench_Pro",
            Self::Multilingual => "princeton-nlp/SWE-bench_Multilingual",
        }
    }
}

#[derive(Debug, Clone)]
struct SweBenchConfig {
    dataset: SweBenchDataset,
    dataset_id: String,
    split: String,
    #[allow(dead_code)]
    // parsed from config but not yet consumed by harness; reserved for sampling support
    num_samples: Option<usize>,
    token_env: Option<String>,
    timeout_secs: u64,
    #[allow(dead_code)] // parsed from config but not yet consumed; reserved for local repo cloning
    host_repo_path: Option<PathBuf>,
    harness_image: String,
    build_images: bool,
    max_workers: usize,
    docker_socket_path: PathBuf,
    mount_docker_socket: bool,
    /// Agent mode: "zero-shot" (default, single prompt) or "bash" (mini-swe-agent style loop)
    #[allow(dead_code)] // consumed by execute_one agent loop path
    agent_mode: AgentMode,
    /// Maximum iterations for bash agent mode
    #[allow(dead_code)] // consumed by execute_one agent loop path
    max_iterations: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct SweBenchInstance {
    repo: String,
    instance_id: String,
    base_commit: String,
    #[serde(default)]
    patch: Option<String>,
    #[serde(default)]
    test_patch: Option<String>,
    problem_statement: String,
    #[serde(default)]
    hints_text: Option<String>,
    #[serde(default)]
    created_at: Option<String>,
    #[serde(default)]
    version: Option<String>,
    #[serde(default, rename = "FAIL_TO_PASS")]
    fail_to_pass: Option<JsonValue>,
    #[serde(default, rename = "PASS_TO_PASS")]
    pass_to_pass: Option<JsonValue>,
    #[serde(default)]
    environment_setup_commit: Option<String>,
    #[serde(default)]
    difficulty: Option<String>,
}

#[allow(dead_code)] // used by harness for prediction format
#[derive(Debug, Serialize)]
struct SweBenchPrediction<'a> {
    instance_id: &'a str,
    model_name_or_path: &'a str,
    model_patch: &'a str,
}

impl Benchmark for SweBenchBenchmark {
    fn name(&self) -> &str {
        "swebench"
    }
    fn display_name(&self) -> &'static str {
        "SWE-Bench"
    }
    fn category(&self) -> BenchmarkCategory {
        BenchmarkCategory::LongContextCoding
    }

    fn pre_execute(&self, config: &yaml_serde::Value) -> Result<()> {
        let cfg = parse_config(SweBenchDataset::Basic, config)?;
        prepare_swebench(&cfg)?;
        let items = load_or_download_dataset(&cfg)?;
        println!("SWE-Bench: {} instances", items.len());
        let mut state = self.state.lock().expect(crate::shared::MUTEX_PANIC_MSG);
        state.items = items;
        state.current_idx = 0;
        state.dataset = SweBenchDataset::Basic;
        state.config = Some(cfg);
        Ok(())
    }

    fn execute_one(
        &self,
        model: &Model,
        _config: &yaml_serde::Value,
        tracker: &mut TokenTracker,
    ) -> Result<Option<TaskResult>> {
        let (instance, idx, cfg) = {
            let mut state = self.state.lock().expect(crate::shared::MUTEX_PANIC_MSG);
            if state.current_idx >= state.items.len() {
                return Ok(None);
            }
            let idx = state.current_idx;
            state.current_idx += 1;
            let item = state.items[idx].clone();
            let cfg = state.config.clone();
            (item, idx, cfg)
        };

        // Run outside lock scope — LLM calls run without holding the mutex
        let (result, patch) = execute_one_impl(
            idx,
            &instance,
            model,
            tracker,
            cfg,
            |inst| serde_json::json!({ "instance_id": inst.instance_id }),
        )?;

        // Briefly lock just for HashMap insert
        if let Some(p) = patch {
            let mut state = self.state.lock().expect(crate::shared::MUTEX_PANIC_MSG);
            state.generated_patches.insert(idx, p);
        }
        Ok(Some(result))
    }

    fn to_report_result(&self, b: &BenchmarkResult) -> Result<BenchmarkResult> {
        // SWE-Bench basic includes total_instances
        Ok(build_swe_bench_report(b, true))
    }

    fn batch_evaluate(
        &self,
        task_results: &[TaskResult],
        config: &yaml_serde::Value,
    ) -> Result<Option<Vec<TaskResult>>> {
        let state = self.state.lock().expect(crate::shared::MUTEX_PANIC_MSG);
        if state.generated_patches.is_empty() {
            return Ok(None);
        }
        batch_evaluate_impl(
            &state.items,
            &state.generated_patches,
            task_results,
            state.dataset,
            "SWE-Bench",
            config,
        )
    }
}

impl Benchmark for SweBenchVerifiedBenchmark {
    fn name(&self) -> &str {
        "swebench_verified"
    }
    fn display_name(&self) -> &'static str {
        "SWE-Bench Verified"
    }
    fn category(&self) -> BenchmarkCategory {
        BenchmarkCategory::LongContextCoding
    }

    fn pre_execute(&self, config: &yaml_serde::Value) -> Result<()> {
        let cfg = parse_config(SweBenchDataset::Verified, config)?;
        prepare_swebench(&cfg)?;
        let items = load_or_download_dataset(&cfg)?;
        println!("SWE-Bench Verified: {} instances", items.len());
        let mut state = self.state.lock().expect(crate::shared::MUTEX_PANIC_MSG);
        state.items = items;
        state.current_idx = 0;
        state.config = Some(cfg);
        Ok(())
    }

    fn execute_one(
        &self,
        model: &Model,
        _config: &yaml_serde::Value,
        tracker: &mut TokenTracker,
    ) -> Result<Option<TaskResult>> {
        let (instance, idx, cfg) = {
            let mut state = self.state.lock().expect(crate::shared::MUTEX_PANIC_MSG);
            if state.current_idx >= state.items.len() {
                return Ok(None);
            }
            let idx = state.current_idx;
            state.current_idx += 1;
            let item = state.items[idx].clone();
            let cfg = state.config.clone();
            (item, idx, cfg)
        };

        // Run outside lock scope — LLM calls run without holding the mutex
        let (result, patch) = execute_one_impl(
            idx,
            &instance,
            model,
            tracker,
            cfg,
            |inst| serde_json::json!({ "instance_id": inst.instance_id }),
        )?;

        // Briefly lock just for HashMap insert
        if let Some(p) = patch {
            let mut state = self.state.lock().expect(crate::shared::MUTEX_PANIC_MSG);
            state.generated_patches.insert(idx, p);
        }
        Ok(Some(result))
    }

    fn batch_evaluate(
        &self,
        task_results: &[TaskResult],
        config: &yaml_serde::Value,
    ) -> Result<Option<Vec<TaskResult>>> {
        let state = self.state.lock().expect(crate::shared::MUTEX_PANIC_MSG);
        if state.generated_patches.is_empty() {
            return Ok(None);
        }
        batch_evaluate_impl(
            &state.items,
            &state.generated_patches,
            task_results,
            state
                .config
                .as_ref()
                .map(|c| c.dataset)
                .unwrap_or(SweBenchDataset::Verified),
            "SWE-Bench Verified",
            config,
        )
    }

    fn to_report_result(&self, b: &BenchmarkResult) -> Result<BenchmarkResult> {
        // SWE-Bench Verified does NOT include total_instances
        Ok(build_swe_bench_report(b, false))
    }
}

impl Benchmark for SweBenchProBenchmark {
    fn name(&self) -> &str {
        "swebench_pro"
    }
    fn display_name(&self) -> &'static str {
        "SWE-Bench Pro"
    }
    fn category(&self) -> BenchmarkCategory {
        BenchmarkCategory::LongContextCoding
    }

    fn pre_execute(&self, config: &yaml_serde::Value) -> Result<()> {
        let cfg = parse_config(SweBenchDataset::Pro, config)?;
        prepare_swebench(&cfg)?;
        let items = load_or_download_dataset(&cfg)?;
        println!("SWE-Bench Pro: {} instances", items.len());
        let mut state = self.state.lock().expect(crate::shared::MUTEX_PANIC_MSG);
        state.items = items;
        state.current_idx = 0;
        state.config = Some(cfg);
        Ok(())
    }

    fn execute_one(
        &self,
        model: &Model,
        _config: &yaml_serde::Value,
        tracker: &mut TokenTracker,
    ) -> Result<Option<TaskResult>> {
        let (instance, idx, cfg) = {
            let mut state = self.state.lock().expect(crate::shared::MUTEX_PANIC_MSG);
            if state.current_idx >= state.items.len() {
                return Ok(None);
            }
            let idx = state.current_idx;
            state.current_idx += 1;
            let item = state.items[idx].clone();
            let cfg = state.config.clone();
            (item, idx, cfg)
        };

        // Run outside lock scope — LLM calls run without holding the mutex
        let (result, patch) = execute_one_impl(
            idx,
            &instance,
            model,
            tracker,
            cfg,
            |inst| serde_json::json!({ "instance_id": inst.instance_id }),
        )?;

        // Briefly lock just for HashMap insert
        if let Some(p) = patch {
            let mut state = self.state.lock().expect(crate::shared::MUTEX_PANIC_MSG);
            state.generated_patches.insert(idx, p);
        }
        Ok(Some(result))
    }

    fn batch_evaluate(
        &self,
        task_results: &[TaskResult],
        config: &yaml_serde::Value,
    ) -> Result<Option<Vec<TaskResult>>> {
        let state = self.state.lock().expect(crate::shared::MUTEX_PANIC_MSG);
        if state.generated_patches.is_empty() {
            return Ok(None);
        }
        batch_evaluate_impl(
            &state.items,
            &state.generated_patches,
            task_results,
            state
                .config
                .as_ref()
                .map(|c| c.dataset)
                .unwrap_or(SweBenchDataset::Pro),
            "SWE-Bench Pro",
            config,
        )
    }

    fn to_report_result(&self, b: &BenchmarkResult) -> Result<BenchmarkResult> {
        // SWE-Bench Pro does NOT include total_instances
        Ok(build_swe_bench_report(b, false))
    }
}

impl Benchmark for SweBenchMultilingualBenchmark {
    fn name(&self) -> &str {
        "swebench_multilingual"
    }
    fn display_name(&self) -> &'static str {
        "SWE-Bench Multilingual"
    }
    fn category(&self) -> BenchmarkCategory {
        BenchmarkCategory::LongContextCoding
    }

    fn pre_execute(&self, config: &yaml_serde::Value) -> Result<()> {
        let cfg = parse_config(SweBenchDataset::Multilingual, config)?;
        prepare_swebench(&cfg)?;
        let items = load_or_download_dataset(&cfg)?;
        println!("SWE-Bench Multilingual: {} instances", items.len());
        let mut state = self.state.lock().expect(crate::shared::MUTEX_PANIC_MSG);
        state.items = items;
        state.current_idx = 0;
        state.config = Some(cfg);
        Ok(())
    }

    fn execute_one(
        &self,
        model: &Model,
        _config: &yaml_serde::Value,
        tracker: &mut TokenTracker,
    ) -> Result<Option<TaskResult>> {
        let (instance, idx, cfg) = {
            let mut state = self.state.lock().expect(crate::shared::MUTEX_PANIC_MSG);
            if state.current_idx >= state.items.len() {
                return Ok(None);
            }
            let idx = state.current_idx;
            state.current_idx += 1;
            let item = state.items[idx].clone();
            let cfg = state.config.clone();
            (item, idx, cfg)
        };

        // Run outside lock scope — LLM calls run without holding the mutex
        let (result, patch) = execute_one_impl(idx, &instance, model, tracker, cfg, |inst| {
            serde_json::json!({
                "instance_id": inst.instance_id,
                "repo": inst.repo,
                "language": get_language_for_repo(&inst.repo),
            })
        })?;

        // Briefly lock just for HashMap insert
        if let Some(p) = patch {
            let mut state = self.state.lock().expect(crate::shared::MUTEX_PANIC_MSG);
            state.generated_patches.insert(idx, p);
        }
        Ok(Some(result))
    }

    fn batch_evaluate(
        &self,
        task_results: &[TaskResult],
        config: &yaml_serde::Value,
    ) -> Result<Option<Vec<TaskResult>>> {
        let state = self.state.lock().expect(crate::shared::MUTEX_PANIC_MSG);
        if state.generated_patches.is_empty() {
            return Ok(None);
        }
        batch_evaluate_impl(
            &state.items,
            &state.generated_patches,
            task_results,
            state
                .config
                .as_ref()
                .map(|c| c.dataset)
                .unwrap_or(SweBenchDataset::Multilingual),
            "SWE-Bench Multilingual",
            config,
        )
    }

    fn to_report_result(&self, b: &BenchmarkResult) -> Result<BenchmarkResult> {
        // Start with the common SWE-Bench report including total_instances
        let mut result = build_swe_bench_report(b, true);

        // Per-language breakdown table
        if let Some(per_task) = b.raw.get("per_task").and_then(|v| v.as_array()) {
            let mut lang_counts: BTreeMap<String, (i64, i64)> = BTreeMap::new();

            for task in per_task {
                let language = task
                    .get("metadata")
                    .and_then(|m| m.get("language").and_then(|v| v.as_str()))
                    .unwrap_or("Unknown")
                    .to_string();
                let passed = task
                    .get("passed")
                    .and_then(|v| v.as_bool())
                    .unwrap_or(false);

                let (p, t) = lang_counts.entry(language).or_insert((0, 0));
                *t += 1;
                if passed {
                    *p += 1;
                }
            }

            if !lang_counts.is_empty() {
                let mut rows = BTreeMap::new();
                for (language, (passed, total_lang)) in &lang_counts {
                    let rate = if *total_lang > 0 {
                        *passed as f64 / *total_lang as f64
                    } else {
                        0.0
                    };
                    rows.insert(
                        language.clone(),
                        BTreeMap::from([
                            (
                                "pass_rate".to_string(),
                                Score::float(rate, ScoreUnit::Percent)
                                    .display(format!("{:.1}%", rate * 100.0)),
                            ),
                            (
                                "instances".to_string(),
                                Score::integer(*total_lang, ScoreUnit::Count)
                                    .display(format!("{}/{}", passed, total_lang)),
                            ),
                        ]),
                    );
                }
                result.breakdowns.insert(
                    "By Language".to_string(),
                    BreakdownTable {
                        title: "Pass Rate by Programming Language".to_string(),
                        rows,
                    },
                );
            }
        }

        Ok(result)
    }
}

#[derive(Debug, Clone)]
struct HarnessResult {
    passed: bool,
    timed_out: bool,
    #[allow(dead_code)] // captured for future diagnostics
    exit_code: Option<i32>,
    #[allow(dead_code)] // captured for future diagnostics
    stdout: String,
    #[allow(dead_code)] // captured for future diagnostics
    stderr: String,
    error_summary: String,
}

fn prepare_swebench(cfg: &SweBenchConfig) -> Result<()> {
    ensure_swebench_harness_image(cfg)?;
    let _ = load_or_download_dataset(cfg)?;
    Ok(())
}

/// Build a SWE-Bench report result from an in-memory BenchmarkResult.
/// If `include_total_instances` is true, adds a "total_instances" score.
fn build_swe_bench_report(b: &BenchmarkResult, include_total_instances: bool) -> BenchmarkResult {
    let raw = &b.raw;
    let (total, resolved, output_tokens, thinking_tokens) = {
        if let Some(per_task) = raw.get("per_task").and_then(|v| v.as_array()) {
            let total = per_task.len() as i64;
            let resolved = per_task
                .iter()
                .filter(|t| t.get("passed").and_then(|v| v.as_bool()).unwrap_or(false))
                .count() as i64;
            let out: u64 = per_task
                .iter()
                .filter_map(|t| t.get("output_tokens").and_then(|v| v.as_u64()))
                .sum();
            let think: u64 = per_task
                .iter()
                .filter_map(|t| t.get("thinking_tokens").and_then(|v| v.as_u64()))
                .sum();
            (total, resolved, out, think)
        } else {
            (
                raw.get("total_instances")
                    .and_then(|v| v.as_i64())
                    .unwrap_or(0),
                raw.get("resolved").and_then(|v| v.as_i64()).unwrap_or(0),
                raw.get("output_tokens")
                    .and_then(|v| v.as_u64())
                    .unwrap_or(0),
                raw.get("thinking_tokens")
                    .and_then(|v| v.as_u64())
                    .unwrap_or(0),
            )
        }
    };
    let pass_rate = if total > 0 {
        resolved as f64 / total as f64 * 100.0
    } else {
        0.0
    };
    let mut scores = BTreeMap::new();
    scores.insert(
        "pass_rate".to_string(),
        Score::float(pass_rate, ScoreUnit::Percent)
            .primary(true)
            .higher_is_better(true),
    );
    scores.insert(
        "resolved".to_string(),
        Score::integer(resolved, ScoreUnit::Count),
    );
    if include_total_instances {
        scores.insert(
            "total_instances".to_string(),
            Score::integer(total, ScoreUnit::Count),
        );
    }
    if output_tokens > 0 {
        scores.insert(
            "output_tokens".to_string(),
            Score::integer(output_tokens as i64, ScoreUnit::Tokens),
        );
    }
    if thinking_tokens > 0 {
        scores.insert(
            "thinking_tokens".to_string(),
            Score::integer(thinking_tokens as i64, ScoreUnit::Tokens),
        );
    }
    BenchmarkResult {
        scores,
        breakdowns: BTreeMap::new(),
        error_classification: BTreeMap::new(),
        artifacts: vec![],
        diagnostics: vec![],
        raw: raw.clone(),
    }
}

fn ensure_swebench_harness_image(cfg: &SweBenchConfig) -> Result<()> {
    if !cfg.build_images {
        return Ok(());
    }
    if DockerRunner::image_exists(&cfg.harness_image)? {
        return Ok(());
    }

    let context = Path::new("docker").join("swebench-harness");
    let dockerfile = context.join("Dockerfile");
    if !dockerfile.exists() {
        return Err(anyhow::anyhow!(
            "docker.build_images=true, but SWE-Bench harness Dockerfile was not found at {}",
            dockerfile.display()
        ));
    }

    println!(
        "  Building SWE-Bench harness image {}...",
        cfg.harness_image
    );
    let build = DockerBuildConfig::new(&cfg.harness_image, dockerfile, context, 300);
    let output = DockerRunner::build_image(&build)?;
    if output.timed_out {
        return Err(anyhow::anyhow!(
            "timed out building SWE-Bench harness image {} after {} seconds",
            cfg.harness_image,
            build.timeout_secs
        ));
    }
    if !output.success() {
        return Err(anyhow::anyhow!(
            "failed to build SWE-Bench harness image {} (exit {:?})\nstdout:\n{}\nstderr:\n{}",
            cfg.harness_image,
            output.exit_code,
            truncate(&output.stdout, 4000),
            truncate(&output.stderr, 4000)
        ));
    }
    Ok(())
}

// Run the swebench-harness Docker container for batch patch evaluation
fn run_swebench_harness(
    cfg: &SweBenchConfig,
    run_dir: &Path,
    predictions_path: &Path,
) -> Result<HarnessResult> {
    let mut docker = DockerRunConfig::new(
        &cfg.harness_image,
        vec![
            "python".to_string(),
            "-m".to_string(),
            "swebench.harness.run_evaluation".to_string(),
            "--dataset_name".to_string(),
            cfg.dataset_id.clone(),
            "--split".to_string(),
            cfg.split.clone(),
            "--predictions_path".to_string(),
            "/work/predictions.jsonl".to_string(),
            "--max_workers".to_string(),
            cfg.max_workers.to_string(),
            "--run_id".to_string(),
            cfg.dataset.benchmark_name().to_string(),
        ],
        cfg.timeout_secs,
    );
    docker.mounts.push(DockerMount::readwrite(run_dir, "/work"));
    if cfg.mount_docker_socket {
        eprintln!(
            "  WARNING: Docker socket mounted — harness container has full Docker access on host"
        );
        if !cfg.docker_socket_path.exists() {
            return Err(anyhow::anyhow!(
                "SWE-Bench official harness requires Docker socket access, but {} does not exist. Set docker.mount_docker_socket=false only for harness images that do not need Docker, or configure docker.docker_socket_path.",
                cfg.docker_socket_path.display()
            ));
        }
        docker.mounts.push(DockerMount::direct_readonly(
            &cfg.docker_socket_path,
            "/var/run/docker.sock",
        ));
    }
    docker.workdir = Some("/work".to_string());
    docker.host_repo_path = cfg.host_repo_path.clone();
    docker.name_prefix = "llm-benchmark-runner-swebench".to_string();
    // SWE-Bench harness needs network for repository/image/dataset setup unless all artifacts are pre-cached.
    docker.network_none = false;
    docker.read_only_root = true;
    docker.tmpfs = vec![
        "/tmp".to_string(),
        "/var/run/docker.sock".to_string(),
        "/root".to_string(),
    ];
    docker.pids_limit = Some(200);
    docker.memory = Some("4g".to_string());

    // Whitelist allowed token environment variable names for security
    const ALLOWED_TOKEN_ENVS: &[&str] = &[
        "HF_TOKEN",
        "GH_TOKEN",
        "HUGGING_FACE_HUB_TOKEN",
        "GITHUB_TOKEN",
    ];

    if let Some(token_env) = &cfg.token_env {
        if !ALLOWED_TOKEN_ENVS.contains(&token_env.as_str()) {
            return Err(anyhow::anyhow!(
                "Refusing to pass env var '{}' to Docker container. \
                 Allowed token env vars: {:?}",
                token_env,
                ALLOWED_TOKEN_ENVS
            ));
        }
        if let Ok(token) = std::env::var(token_env) {
            eprintln!(
                "  NOTE: Token env '{}' passed to container (visible via 'docker inspect')",
                token_env
            );
            docker.env.push((token_env.clone(), token.clone()));
            if token_env != "HF_TOKEN" {
                docker.env.push(("HF_TOKEN".to_string(), token));
            }
        }
    }
    if let Some(parent) = predictions_path.parent() {
        fs::create_dir_all(parent)?;
    }

    let out = DockerRunner::run(&docker)?;
    let passed = out.success();
    Ok(HarnessResult {
        passed,
        timed_out: out.timed_out,
        exit_code: out.exit_code,
        error_summary: if out.timed_out {
            format!("timed out after {} seconds", cfg.timeout_secs)
        } else if passed {
            "harness completed".to_string()
        } else {
            truncate(&out.stderr, 1200)
        },
        stdout: out.stdout,
        stderr: out.stderr,
    })
}

/// Parse resolved instance IDs from SWE-Bench harness output JSON files.
fn parse_harness_results(run_dir: &Path) -> Result<HashSet<String>> {
    let mut json_files = Vec::new();
    collect_json_files(run_dir, &mut json_files, 0)?;

    let mut resolved_instance_ids: HashSet<String> = HashSet::new();
    for path in &json_files {
        let Ok(content) = fs::read_to_string(path) else {
            continue;
        };
        let Ok(value) = serde_json::from_str::<JsonValue>(&content) else {
            continue;
        };

        // Check for resolved_ids arrays
        for key in ["resolved_ids", "resolved", "resolved_instances"] {
            if let Some(arr) = value.get(key).and_then(|v| v.as_array()) {
                for id in arr {
                    if let Some(s) = id.as_str() {
                        resolved_instance_ids.insert(s.to_string());
                    }
                }
            }
        }

        // Check for per-instance resolved: true fields
        if let Some(obj) = value.as_object() {
            for (k, v) in obj {
                if let Some(true) = v.get("resolved").and_then(|r| r.as_bool()) {
                    resolved_instance_ids.insert(k.clone());
                }
            }
        }
    }

    Ok(resolved_instance_ids)
}

/// Default config for a given SWE-Bench dataset variant.
fn default_config(dataset: SweBenchDataset) -> SweBenchConfig {
    SweBenchConfig {
        dataset,
        dataset_id: String::new(),
        split: String::new(),
        num_samples: None,
        token_env: None,
        timeout_secs: 1800,
        host_repo_path: None,
        harness_image: String::new(),
        build_images: false,
        max_workers: 1,
        docker_socket_path: PathBuf::from("/var/run/docker.sock"),
        mount_docker_socket: false,
        agent_mode: AgentMode::ZeroShot,
        max_iterations: 50.min(MAX_ALLOWED_ITERATIONS),
    }
}

/// Shared batch evaluation logic for all SWE-Bench variants.
fn batch_evaluate_impl(
    items: &[SweBenchInstance],
    generated_patches: &HashMap<usize, String>,
    task_results: &[TaskResult],
    dataset: SweBenchDataset,
    harness_label: &str,
    config: &yaml_serde::Value,
) -> Result<Option<Vec<TaskResult>>> {
    // Build patch_pairs using index-based lookup
    let mut patch_pairs: Vec<(SweBenchInstance, String, String)> = Vec::new();
    for (idx, tr) in task_results.iter().enumerate() {
        if let Some(patch) = generated_patches.get(&idx) {
            if idx < items.len() {
                let instance = &items[idx];
                patch_pairs.push((instance.clone(), patch.clone(), tr.task_id.clone()));
            }
        }
    }

    if patch_pairs.is_empty() {
        return Ok(None);
    }

    // Build O(1) lookup: task_id -> instance
    let task_to_instance: HashMap<&str, &SweBenchInstance> = patch_pairs
        .iter()
        .map(|(instance, _, tid)| (tid.as_str(), instance))
        .collect();

    // Parse harness config
    let cfg = parse_config(dataset, config)?;

    // Create temporary working directory for harness I/O
    let run_tempdir = tempfile::tempdir_in(
        dirs::cache_dir()
            .unwrap_or_else(|| PathBuf::from("/tmp"))
            .join("llm-benchmark-runner"),
    )
    .context("failed to create temp dir for SWE-Bench harness")?;
    let run_dir = run_tempdir.path();
    let predictions_path = run_dir.join("predictions.jsonl");

    // Write predictions.jsonl
    {
        let mut file = fs::File::create(&predictions_path)?;
        for (instance, patch, _tid) in &patch_pairs {
            let prediction = serde_json::json!({
                "instance_id": instance.instance_id,
                "model_name_or_path": "llm-benchmark-runner",
                "model_patch": patch,
            });
            writeln!(file, "{}", serde_json::to_string(&prediction)?)?;
        }
    }
    println!(
        "  Running {} harness for {} patches...",
        harness_label,
        patch_pairs.len()
    );

    // Run the actual SWE-Bench harness
    let harness_result = run_swebench_harness(&cfg, run_dir, &predictions_path)?;

    if harness_result.timed_out {
        eprintln!(
            "  Warning: {} harness timed out after {} seconds",
            harness_label, cfg.timeout_secs
        );
    }
    if !harness_result.passed {
        eprintln!(
            "  Warning: {} harness exited with error: {}",
            harness_label, harness_result.error_summary
        );
    }

    // Parse resolved instance IDs from harness output
    let resolved_instance_ids = parse_harness_results(run_dir)?;

    println!(
        "  {} harness: {} resolved / {} total ({:.1}%)",
        harness_label,
        resolved_instance_ids.len(),
        patch_pairs.len(),
        if patch_pairs.is_empty() {
            0.0
        } else {
            resolved_instance_ids.len() as f64 / patch_pairs.len() as f64 * 100.0
        }
    );

    // Map harness results back to TaskResults
    let updated: Vec<TaskResult> = task_results
        .iter()
        .map(|tr| {
            let mut updated = tr.clone();
            if let Some(instance) = task_to_instance.get(tr.task_id.as_str()) {
                updated.passed = resolved_instance_ids.contains(&instance.instance_id);
                updated.score = if updated.passed { 1.0 } else { 0.0 };
            }
            updated
        })
        .collect();

    Ok(Some(updated))
}

/// Shared execute_one logic for all SWE-Bench variants.
/// Returns (TaskResult, Option<patch_string>). The patch is None if validation failed.
fn execute_one_impl(
    idx: usize,
    instance: &SweBenchInstance,
    model: &Model,
    tracker: &mut TokenTracker,
    config: Option<SweBenchConfig>,
    metadata_builder: impl FnOnce(&SweBenchInstance) -> serde_json::Value,
) -> Result<(TaskResult, Option<String>)> {
    let cfg = config.unwrap_or_else(|| default_config(SweBenchDataset::Basic));

    // Choose execution mode — handle agent loop failures gracefully
    let patch = match cfg.agent_mode {
        AgentMode::Bash => match execute_agent_loop(model, instance, &cfg, tracker) {
            Ok(p) => p,
            Err(e) => {
                eprintln!("  Agent loop failed for {}: {}", instance.instance_id, e);
                let task_id = format!("task-{}", idx);
                return Ok((
                    TaskResult::new(task_id, false, 0.0, vec![instance.repo.clone()])
                        .with_metadata(Some(metadata_builder(instance))),
                    None,
                ));
            }
        },
        AgentMode::ZeroShot => execute_zero_shot(model, instance, tracker)?,
    };

    // Validate patch content — if invalid, return failed task with no patch
    if let Err(e) = validate_patch(&patch) {
        eprintln!("  Patch validation failed for task-{}: {}", idx, e);
        let task_id = format!("task-{}", idx);
        return Ok((
            TaskResult::new(task_id, false, 0.0, vec![instance.repo.clone()])
                .with_metadata(Some(metadata_builder(instance))),
            None,
        ));
    }

    // Truncate oversized patches
    let patch = if patch.len() > MAX_PATCH_BYTES {
        eprintln!(
            "  Warning: patch for task-{} exceeds {} bytes ({} bytes), truncating",
            idx,
            MAX_PATCH_BYTES,
            patch.len()
        );
        patch[..MAX_PATCH_BYTES].to_string()
    } else {
        patch
    };

    let task_id = format!("task-{}", idx);
    Ok((
        TaskResult::new(task_id, false, 0.0, vec![instance.repo.clone()])
            .with_metadata(Some(metadata_builder(instance))),
        Some(patch),
    ))
}

fn parse_config(dataset: SweBenchDataset, config: &yaml_serde::Value) -> Result<SweBenchConfig> {
    let docker_cfg = config.get("__docker");
    if docker_cfg
        .and_then(|docker| docker.get("enabled"))
        .and_then(|v| v.as_bool())
        == Some(false)
    {
        return Err(anyhow::anyhow!(
            "Docker is disabled, but SWE-Bench benchmarks require Docker"
        ));
    }
    let dataset_id = config
        .get("dataset_id")
        .and_then(|v| v.as_str())
        .unwrap_or_else(|| dataset.default_dataset_id())
        .to_string();
    let split = config
        .get("split")
        .and_then(|v| v.as_str())
        .unwrap_or("test")
        .to_string();
    let num_samples = config
        .get("num_samples")
        .and_then(|v| v.as_i64())
        .map(|n| n.max(0) as usize);
    let token_env = config
        .get("token_env")
        .and_then(|v| v.as_str())
        .map(ToOwned::to_owned);
    let timeout_secs = config
        .get("timeout_secs")
        .and_then(|v| v.as_i64())
        .unwrap_or(1800)
        .max(1) as u64;
    let host_repo_path = docker_cfg
        .and_then(|docker| docker.get("host_repo_path"))
        .and_then(|v| v.as_str())
        .map(PathBuf::from);
    let harness_image = docker_cfg
        .and_then(|docker| docker.get("images"))
        .and_then(|images| images.get("swebench_harness"))
        .and_then(|v| v.as_str())
        .unwrap_or("llm-benchmark-runner/swebench-harness:latest")
        .to_string();
    let build_images = docker_cfg
        .and_then(|docker| docker.get("build_images"))
        .and_then(|v| v.as_bool())
        .unwrap_or(false);
    let max_workers = docker_cfg
        .and_then(|docker| docker.get("max_workers"))
        .and_then(|v| v.as_i64())
        .unwrap_or(1)
        .max(1) as usize;
    let docker_socket_path = docker_cfg
        .and_then(|docker| docker.get("docker_socket_path"))
        .and_then(|v| v.as_str())
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("/var/run/docker.sock"));
    let mount_docker_socket = docker_cfg
        .and_then(|docker| docker.get("mount_docker_socket"))
        .and_then(|v| v.as_bool())
        .unwrap_or(false);
    let agent_mode: AgentMode = config
        .get("agent_mode")
        .and_then(|v| v.as_str())
        .unwrap_or("zero-shot")
        .parse()
        .context("Invalid agent_mode: use 'zero-shot' or 'bash'")?;
    let max_iterations = config
        .get("max_iterations")
        .and_then(|v| v.as_i64())
        .unwrap_or(50)
        .max(1) as usize;
    let max_iterations = if max_iterations > MAX_ALLOWED_ITERATIONS {
        eprintln!(
            "  Warning: max_iterations {} exceeds maximum of {}, clamping",
            max_iterations, MAX_ALLOWED_ITERATIONS
        );
        MAX_ALLOWED_ITERATIONS
    } else {
        max_iterations
    };
    Ok(SweBenchConfig {
        dataset,
        dataset_id,
        split,
        num_samples,
        token_env,
        timeout_secs,
        host_repo_path,
        harness_image,
        build_images,
        max_workers,
        docker_socket_path,
        mount_docker_socket,
        agent_mode,
        max_iterations,
    })
}

fn load_or_download_dataset(cfg: &SweBenchConfig) -> Result<Vec<SweBenchInstance>> {
    let path = dataset_cache_path(cfg)?;
    if path.exists() {
        return read_dataset_jsonl(&path);
    }
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }
    println!(
        "  Downloading {} split {} from HuggingFace...",
        cfg.dataset_id, cfg.split
    );
    let rows = download_hf_rows(cfg)?;
    let tmp = path.with_extension(format!("jsonl.tmp.{}", std::process::id()));
    let mut file = fs::File::create(&tmp)?;
    for row in &rows {
        writeln!(file, "{}", serde_json::to_string(row)?)?;
    }
    fs::rename(&tmp, &path).inspect_err(|_err| {
        let _ = fs::remove_file(&tmp);
    })?;
    Ok(rows)
}

fn dataset_cache_path(cfg: &SweBenchConfig) -> Result<PathBuf> {
    let base = dirs::cache_dir()
        .unwrap_or_default()
        .join("llm-benchmark-runner")
        .join("swe_bench");
    Ok(base.join(format!(
        "{}-{}.jsonl",
        sanitize_path_component(&cfg.dataset_id),
        sanitize_path_component(&cfg.split)
    )))
}

fn read_dataset_jsonl(path: &Path) -> Result<Vec<SweBenchInstance>> {
    let content = fs::read_to_string(path)?;
    content
        .lines()
        .filter(|line| !line.trim().is_empty())
        .map(|line| serde_json::from_str(line).context("invalid SWE-Bench cache row"))
        .collect()
}

fn download_hf_rows(cfg: &SweBenchConfig) -> Result<Vec<SweBenchInstance>> {
    let client = reqwest::blocking::Client::builder().build()?;
    let token = cfg
        .token_env
        .as_deref()
        .and_then(|env_name| std::env::var(env_name).ok());
    let mut rows = Vec::new();
    let mut offset = 0usize;
    let page_size = 100usize;
    loop {
        let url = reqwest::Url::parse_with_params(
            "https://datasets-server.huggingface.co/rows",
            &[
                ("dataset", cfg.dataset_id.as_str()),
                ("config", "default"),
                ("split", cfg.split.as_str()),
                ("offset", &offset.to_string()),
                ("length", &page_size.to_string()),
            ],
        )?;
        let mut req = client.get(url);
        if let Some(token) = &token {
            req = req.bearer_auth(token);
        }
        let response: JsonValue = req.send()?.error_for_status()?.json()?;
        let page = response
            .get("rows")
            .and_then(|v| v.as_array())
            .cloned()
            .unwrap_or_default();
        if page.is_empty() {
            break;
        }
        for item in &page {
            let row = item
                .get("row")
                .cloned()
                .ok_or_else(|| anyhow::anyhow!("missing row in HuggingFace response"))?;
            rows.push(serde_json::from_value(row)?);
        }
        if page.len() < page_size {
            break;
        }
        offset += page_size;
    }
    Ok(rows)
}

// Build patch prompt for an SWE-bench instance (used by harness)
#[allow(dead_code)]
fn build_patch_prompt(instance: &SweBenchInstance) -> String {
    format!(
        "You are solving a SWE-Bench repository issue. Return ONLY a unified diff patch. Do not include markdown fences, explanations, or prose.\n\nRepository: {}\nBase commit: {}\nInstance: {}\n\nProblem statement:\n{}\n\nHints:\n{}\n\nReturn the patch now.",
        instance.repo,
        instance.base_commit,
        instance.instance_id,
        instance.problem_statement,
        instance.hints_text.as_deref().unwrap_or("")
    )
}

fn extract_diff(response: &str) -> String {
    let trimmed = response.trim();
    if let Some(start) = trimmed.find("```") {
        let rest = &trimmed[start + 3..];
        let rest = rest
            .strip_prefix("diff")
            .or_else(|| rest.strip_prefix("patch"))
            .unwrap_or(rest)
            .trim_start_matches(['\n', '\r']);
        if let Some(end) = rest.find("```") {
            return rest[..end].trim().to_string();
        }
    }
    trimmed.to_string()
}

/// Validate that a generated patch has the structure of a git diff.
fn validate_patch(patch: &str) -> Result<()> {
    if patch.trim().is_empty() {
        return Err(anyhow::anyhow!("Generated patch is empty"));
    }
    if patch.len() < 20 {
        return Err(anyhow::anyhow!(
            "Generated patch is too short ({} bytes) to be a valid diff",
            patch.len()
        ));
    }
    // Check for basic unified diff markers in the first portion of the patch
    let header = &patch[..patch.len().min(500)];
    if !header.contains("--- ") || !header.contains("+++") {
        return Err(anyhow::anyhow!(
            "Generated patch does not appear to be a valid unified diff \
             (missing '---'/'+++' markers)"
        ));
    }
    Ok(())
}

/// Zero-shot mode: single prompt, extract diff from response.
fn execute_zero_shot(
    model: &Model,
    instance: &SweBenchInstance,
    tracker: &mut TokenTracker,
) -> Result<String> {
    let system_prompt = "You are a software engineer tasked with fixing a bug. Analyze the issue and generate a git diff patch.";
    let prompt = format!(
        "Issue: {}

{}",
        instance.problem_statement, instance.base_commit
    );
    let response = tracker.chat_completion(&model.model_name, system_prompt, &prompt)?;
    Ok(extract_diff(&response))
}

/// Set up the repository in the agent sandbox container.
fn setup_agent_repo(container: &str, instance: &SweBenchInstance) -> Result<()> {
    // Clone repository — propagate errors
    let clone_cmd = format!(
        "git clone --depth 1 https://github.com/{}.git /repo",
        instance.repo
    );
    DockerRunner::exec(container, &clone_cmd)
        .context(format!("failed to clone {}", instance.repo))?;

    // Checkout base commit — propagate errors
    let checkout_cmd = format!(
        "cd /repo && git fetch --depth 1 origin {} && git checkout {}",
        instance.base_commit, instance.base_commit
    );
    DockerRunner::exec(container, &checkout_cmd).context(format!(
        "failed to checkout {} in {}",
        instance.base_commit, instance.repo
    ))?;

    // Verify repository was set up correctly
    let verify_result = DockerRunner::exec(container, "test -d /repo/.git && echo ok")
        .context("failed to verify repository setup")?;
    if !verify_result.trim().ends_with("ok") {
        return Err(anyhow::anyhow!(
            "repository verification failed: /repo/.git does not appear to exist"
        ));
    }

    Ok(())
}

fn execute_agent_loop(
    model: &Model,
    instance: &SweBenchInstance,
    cfg: &SweBenchConfig,
    tracker: &mut TokenTracker,
) -> Result<String> {
    eprintln!(
        "  SWE-Bench agent loop: {} (max {} iterations)",
        instance.instance_id, cfg.max_iterations
    );

    // Start a sandbox container for this instance
    let container_name = format!(
        "llm-bench-agent-{}-{}",
        instance.instance_id.replace(['/', '-'], "_"),
        std::process::id()
    );

    let mut docker = DockerRunConfig::new(
        &cfg.harness_image,
        vec!["sleep".to_string(), "3600".to_string()], // keep alive
        cfg.timeout_secs,
    );
    docker.read_only_root = true;
    docker.network_none = true;
    docker.cap_drop_all = true;
    docker.no_new_privileges = true;
    docker.pids_limit = Some(64);
    docker.memory = Some("256m".to_string());
    docker.tmpfs = vec!["/tmp:rw,noexec,nosuid,size=32m".to_string()];
    docker.name_prefix = container_name.clone();

    let container =
        DockerRunner::run_detached(&docker).context("Failed to start agent sandbox container")?;

    // Ensure cleanup even on error
    struct ContainerGuard(String);
    impl Drop for ContainerGuard {
        fn drop(&mut self) {
            let _ = DockerRunner::stop(&self.0);
        }
    }
    let _guard = ContainerGuard(container.clone());

    // Set up the repository in the container
    setup_agent_repo(&container, instance)?;

    // Bash tool definition
    let bash_tool = serde_json::json!({
        "type": "function",
        "function": {
            "name": "bash",
            "description": "Execute a bash command in the repository environment. Working directory is the repository root. Use this to explore the codebase, search for relevant files, read code, and make edits.",
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {
                        "type": "string",
                        "description": "The bash command to execute."
                    }
                },
                "required": ["command"]
            }
        }
    });

    let system_prompt = format!(
        "You are an AI software engineer. You have access to a repository and can execute bash commands.

The issue you need to fix:
{}

Repository: {}
Base commit: {}

Use commands like `ls`, `cat <file>`, `grep -r \"<pattern>\" .`, `find .`, `git log`, `git diff`.
When you've identified the fix, use `git diff` to generate a patch and output it in a ```diff code block.
If the fix is simple, you can use `sed` or write files directly.

You have {} iterations maximum. Be efficient with your commands.",
        instance.problem_statement,
        instance.repo,
        instance.base_commit,
        cfg.max_iterations
    );

    // Initial user message
    let initial_prompt =
        "Starting exploration. First, run `ls -la` to see the repository structure, \
         then find relevant files for the issue above. Use bash commands to explore."
            .to_string();

    let mut last_response = String::new();

    for iteration in 0..cfg.max_iterations {
        // Get model response with tools
        let (response, tool_calls) = tracker.chat_completion_with_tools(
            &model.model_name,
            &system_prompt,
            if iteration == 0 {
                &initial_prompt
            } else {
                "Continue. Execute the next bash command or generate a patch if ready."
            },
            vec![bash_tool.clone()],
            None,
            true, // use_history
        )?;

        last_response = response.clone();

        // Check if response contains a diff patch
        let diff = extract_diff(&response);
        if diff.len() > 100 && diff.starts_with("--- ") {
            eprintln!(
                "  Agent {} iteration {}: patch generated ({} bytes)",
                instance.instance_id,
                iteration,
                diff.len()
            );
            return Ok(diff);
        }

        // Process tool calls with REAL execution
        if !tool_calls.is_empty() {
            for tc in &tool_calls {
                // Parse bash command from tool call
                let command = tc
                    .arguments
                    .get("command")
                    .and_then(|v| v.as_str())
                    .unwrap_or("echo 'no command'");

                eprintln!(
                    "  Agent {} iteration {}: bash > {}",
                    instance.instance_id,
                    iteration,
                    command.chars().take(80).collect::<String>()
                );

                // Execute command (runs in Docker sandbox — cap_drop=ALL, network_none, read-only root)
                let tool_result = match DockerRunner::exec(&container, command) {
                    Ok(output) => {
                        let truncated = truncate(&output, 8192);
                        format!(
                            "[Command succeeded, {} bytes output]\n{}",
                            output.len(),
                            truncated
                        )
                    }
                    Err(e) => format!("[Command failed: {}]", e),
                };

                tracker.append_tool_result(&tc.id, &tool_result);
            }
        } else {
            // No tool calls — just continue
            eprintln!(
                "  Agent {} iteration {}: no tool calls, continuing...",
                instance.instance_id, iteration
            );
        }
    }

    // Max iterations reached — try git diff from container as fallback
    eprintln!(
        "  Agent {}: max iterations ({}) reached, extracting best patch",
        instance.instance_id, cfg.max_iterations
    );

    // Try git diff from container as first fallback
    if let Ok(container_diff) = DockerRunner::exec(&container, "git -C /repo diff") {
        if container_diff.len() > 100 && container_diff.starts_with("--- ") {
            return Ok(container_diff);
        }
    }

    // Try extracting diff from last response
    let diff = extract_diff(&last_response);
    if diff.len() > 100 && diff.starts_with("--- ") {
        return Ok(diff);
    }

    // No valid patch produced — return error so execute_one_impl handles gracefully
    Err(anyhow::anyhow!(
        "Agent {} did not produce a valid patch after {} iterations",
        instance.instance_id,
        cfg.max_iterations
    ))
}

#[allow(dead_code)] // kept for future debugging of harness output
fn parse_resolved_count_from_files(json_files: &[PathBuf]) -> Option<usize> {
    let mut best = None;
    for path in json_files {
        let Ok(content) = fs::read_to_string(path) else {
            continue;
        };
        let Ok(value) = serde_json::from_str::<JsonValue>(&content) else {
            continue;
        };
        if let Some(count) = resolved_count_from_json(&value) {
            best = Some(best.map_or(count, |current: usize| current.max(count)));
        }
    }
    best
}

// Recursively collect JSON files from harness output
fn collect_json_files(dir: &Path, out: &mut Vec<PathBuf>, depth: usize) -> Result<()> {
    if depth > 6 || out.len() > 2000 || !dir.is_dir() {
        return Ok(());
    }
    for entry in fs::read_dir(dir)? {
        let entry = entry?;
        let path = entry.path();
        if path.is_dir() {
            collect_json_files(&path, out, depth + 1)?;
        } else if path.extension().and_then(|ext| ext.to_str()) == Some("json") {
            out.push(path);
        }
    }
    Ok(())
}

// Parse resolved count from a single JSON result file
#[allow(dead_code)] // kept for future debugging of harness output
fn resolved_count_from_json(value: &JsonValue) -> Option<usize> {
    if let Some(n) = value.get("resolved").and_then(|v| v.as_u64()) {
        return Some(n as usize);
    }
    for key in ["resolved", "resolved_ids", "resolved_instances"] {
        if let Some(arr) = value.get(key).and_then(|v| v.as_array()) {
            return Some(arr.len());
        }
    }
    if let Some(obj) = value.as_object() {
        let per_instance_resolved = obj
            .values()
            .filter(|entry| {
                entry
                    .get("resolved")
                    .and_then(|v| v.as_bool())
                    .unwrap_or(false)
            })
            .count();
        if per_instance_resolved > 0 {
            return Some(per_instance_resolved);
        }
        for nested_key in ["summary", "results", "evaluation_results"] {
            if let Some(count) = obj.get(nested_key).and_then(resolved_count_from_json) {
                return Some(count);
            }
        }
    }
    None
}

/// Map from repository name to programming language for SWE-bench Multilingual.
/// Based on the official dataset documentation at https://www.swebench.com/multilingual.html
fn get_language_for_repo(repo: &str) -> &'static str {
    REPO_TO_LANGUAGE
        .iter()
        .find_map(|(r, lang)| if *r == repo { Some(*lang) } else { None })
        .unwrap_or("Unknown")
}

// Format: (repo, language)
// Sorted by language then repo for readability
#[rustfmt::skip]
const REPO_TO_LANGUAGE: [(&str, &str); 41] = [
    // C
    ("jqlang/jq", "C"),
    ("micropython/micropython", "C"),
    ("redis/redis", "C"),
    ("valkey-io/valkey", "C"),
    // C++
    ("fmtlib/fmt", "C++"),
    ("nlohmann/json", "C++"),
    // Go
    ("caddyserver/caddy", "Go"),
    ("gin-gonic/gin", "Go"),
    ("gohugoio/hugo", "Go"),
    ("hashicorp/terraform", "Go"),
    ("prometheus/prometheus", "Go"),
    // Java
    ("apache/druid", "Java"),
    ("apache/lucene", "Java"),
    ("google/gson", "Java"),
    ("javaparser/javaparser", "Java"),
    ("projectlombok/lombok", "Java"),
    ("reactivex/rxjava", "Java"),
    // JavaScript/TypeScript
    ("axios/axios", "JavaScript"),
    ("babel/babel", "JavaScript"),
    ("facebook/docusaurus", "JavaScript"),
    ("immutable-js/immutable-js", "JavaScript"),
    ("mrdoob/three.js", "JavaScript"),
    ("preactjs/preact", "JavaScript"),
    ("vuejs/core", "JavaScript"),
    // PHP
    ("briannesbitt/carbon", "PHP"),
    ("laravel/framework", "PHP"),
    ("php-cs-fixer/php-cs-fixer", "PHP"),
    ("phpoffice/phpspreadsheet", "PHP"),
    // Ruby
    ("faker-ruby/faker", "Ruby"),
    ("fastlane/fastlane", "Ruby"),
    ("fluent/fluentd", "Ruby"),
    ("jordansissel/fpm", "Ruby"),
    ("jekyll/jekyll", "Ruby"),
    ("rubocop/rubocop", "Ruby"),
    // Rust
    ("astral-sh/ruff", "Rust"),
    ("burntsushi/ripgrep", "Rust"),
    ("nushell/nushell", "Rust"),
    ("sharkdp/bat", "Rust"),
    ("tokio-rs/axum", "Rust"),
    ("tokio-rs/tokio", "Rust"),
    ("uutils/coreutils", "Rust"),
];

fn sanitize_path_component(value: &str) -> String {
    value
        .chars()
        .map(|c| {
            if c.is_ascii_alphanumeric() || c == '-' || c == '_' || c == '.' {
                c
            } else {
                '_'
            }
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn extracts_fenced_diff() {
        let diff = extract_diff("```diff\ndiff --git a/a b/a\n--- a/a\n+++ b/a\n```");
        assert!(diff.starts_with("diff --git"));
        assert!(!diff.contains("```"));
    }

    #[test]
    fn default_verified_dataset_id_is_public_verified_dataset() {
        assert_eq!(
            SweBenchDataset::Verified.default_dataset_id(),
            "princeton-nlp/SWE-bench_Verified"
        );
    }

    #[test]
    fn resolved_count_parses_nested_official_like_results() {
        let value = serde_json::json!({
            "astropy__astropy-1": {"resolved": true},
            "astropy__astropy-2": {"resolved": false},
            "django__django-1": {"resolved": true}
        });
        assert_eq!(resolved_count_from_json(&value), Some(2));
    }

    #[test]
    fn swebench_timeout_does_not_inherit_short_docker_default() {
        let cfg: yaml_serde::Value = yaml_serde::from_str(
            r#"
    __docker:
      default_timeout_secs: 8
      images:
        swebench_harness: harness:latest
    "#,
        )
        .unwrap();
        let parsed = parse_config(SweBenchDataset::Verified, &cfg).unwrap();
        assert_eq!(parsed.timeout_secs, 1800);
    }

    #[test]
    fn swebench_honors_shared_build_images_flag() {
        let cfg: yaml_serde::Value = yaml_serde::from_str(
            r#"
    __docker:
      build_images: true
      images:
        swebench_harness: harness:latest
    "#,
        )
        .unwrap();
        let parsed = parse_config(SweBenchDataset::Verified, &cfg).unwrap();
        assert!(parsed.build_images);
        assert_eq!(parsed.harness_image, "harness:latest");
    }

    #[test]
    fn default_multilingual_dataset_id() {
        assert_eq!(
            SweBenchDataset::Multilingual.default_dataset_id(),
            "princeton-nlp/SWE-bench_Multilingual"
        );
    }

    #[test]
    fn multilingual_benchmark_name() {
        assert_eq!(
            SweBenchDataset::Multilingual.benchmark_name(),
            "swebench_multilingual"
        );
    }

    #[test]
    fn multilingual_config_parses_correctly() {
        let cfg: yaml_serde::Value = yaml_serde::from_str(
            r#"
    num_samples: 50
    split: test
    timeout_secs: 3600
    __docker:
      images:
        swebench_harness: my-harness:latest
    "#,
        )
        .unwrap();
        let parsed = parse_config(SweBenchDataset::Multilingual, &cfg).unwrap();
        assert_eq!(parsed.split, "test");
        assert_eq!(parsed.timeout_secs, 3600);
        assert_eq!(parsed.dataset_id, "princeton-nlp/SWE-bench_Multilingual");
    }

    #[test]
    fn language_mapping_covers_all_multilingual_repos() {
        // Verify the mapping has the expected number of repos
        assert_eq!(REPO_TO_LANGUAGE.len(), 41);

        // Spot-check a few well-known repos
        assert_eq!(get_language_for_repo("redis/redis"), "C");
        assert_eq!(get_language_for_repo("tokio-rs/tokio"), "Rust");
        assert_eq!(get_language_for_repo("laravel/framework"), "PHP");
        assert_eq!(get_language_for_repo("vuejs/core"), "JavaScript");
        assert_eq!(get_language_for_repo("caddyserver/caddy"), "Go");
        assert_eq!(get_language_for_repo("nlohmann/json"), "C++");
        assert_eq!(get_language_for_repo("projectlombok/lombok"), "Java");
        assert_eq!(get_language_for_repo("rubocop/rubocop"), "Ruby");

        // Unknown repos should return "Unknown"
        assert_eq!(get_language_for_repo("unknown/repo"), "Unknown");
    }

    #[test]
    fn language_mapping_has_correct_language_distribution() {
        let mut counts: BTreeMap<&str, usize> = BTreeMap::new();
        for (_, lang) in &REPO_TO_LANGUAGE {
            *counts.entry(*lang).or_insert(0) += 1;
        }

        // Verify language counts match the dataset
        assert_eq!(counts.get("C"), Some(&4));
        assert_eq!(counts.get("C++"), Some(&2));
        assert_eq!(counts.get("Go"), Some(&5));
        assert_eq!(counts.get("Java"), Some(&6));
        assert_eq!(counts.get("JavaScript"), Some(&7));
        assert_eq!(counts.get("PHP"), Some(&4));
        assert_eq!(counts.get("Ruby"), Some(&6));
        assert_eq!(counts.get("Rust"), Some(&7));
    }

    #[test]
    fn validate_patch_rejects_empty() {
        assert!(validate_patch("").is_err());
        assert!(validate_patch("   \n  ").is_err());
    }

    #[test]
    fn validate_patch_rejects_too_short() {
        assert!(validate_patch("short").is_err());
    }

    #[test]
    fn validate_patch_rejects_non_diff() {
        assert!(validate_patch("This is not a diff\nbut looks long enough for the check").is_err());
    }

    #[test]
    fn validate_patch_accepts_valid_diff() {
        let diff = "--- a/file.py\n+++ b/file.py\n@@ -1 +1 @@\n-old\n+new";
        assert!(validate_patch(diff).is_ok());
    }

    #[test]
    fn extract_diff_from_markdown_fence() {
        let response =
            "Here's the fix:\n```diff\n--- a/file.py\n+++ b/file.py\n@@ -1 +1 @@\n-old\n+new\n```";
        let diff = extract_diff(response);
        assert!(diff.contains("---"));
        assert!(diff.contains("+++"));
        assert!(!diff.contains("```"));
    }

    #[test]
    fn extract_diff_from_generic_fence() {
        let response = "```\n--- a/file.py\n+++ b/file.py\n@@ -1 +1 @@\n-old\n+new\n```";
        let diff = extract_diff(response);
        assert!(diff.contains("---"));
        assert!(!diff.contains("```"));
    }

    #[test]
    fn extract_diff_returns_input_when_no_fence() {
        let diff = "--- a/file.py\n+++ b/file.py\n@@ -1 +1 @@\n-old\n+new";
        assert_eq!(extract_diff(diff), diff);
    }

    #[test]
    fn agent_mode_from_str_variants() {
        assert_eq!(
            "zero-shot".parse::<AgentMode>().unwrap(),
            AgentMode::ZeroShot
        );
        assert_eq!(
            "zero_shot".parse::<AgentMode>().unwrap(),
            AgentMode::ZeroShot
        );
        assert_eq!("bash".parse::<AgentMode>().unwrap(), AgentMode::Bash);
        assert!("invalid".parse::<AgentMode>().is_err());
    }

    #[test]
    fn agent_mode_default_is_zero_shot() {
        assert_eq!(AgentMode::default(), AgentMode::ZeroShot);
    }
}
