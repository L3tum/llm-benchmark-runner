use crate::client::Client;
use crate::config::Model;
use crate::docker_runner::{DockerBuildConfig, DockerMount, DockerRunConfig, DockerRunner};
use crate::reports::model::BenchmarkResult;
use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use serde_json::Value as JsonValue;
use std::collections::BTreeMap;
use std::fs;
use std::io::Write;
use std::path::{Path, PathBuf};

pub struct SweBenchBenchmark;
pub struct SweBenchVerifiedBenchmark;
pub struct SweBenchProBenchmark;

#[derive(Debug, Clone, Copy)]
enum SweBenchDataset {
    Basic,
    Verified,
    Pro,
}

impl SweBenchDataset {
    fn benchmark_name(self) -> &'static str {
        match self {
            Self::Basic => "swebench",
            Self::Verified => "swebench_verified",
            Self::Pro => "swebench_pro",
        }
    }

    fn default_dataset_id(self) -> &'static str {
        match self {
            Self::Basic => "princeton-nlp/SWE-bench",
            Self::Verified => "princeton-nlp/SWE-bench_Verified",
            // Pro access/naming can be gated; users can override with benchmark.swebench_pro.dataset_id.
            Self::Pro => "SWE-bench/SWE-bench_Pro",
        }
    }
}

#[derive(Debug, Clone)]
struct SweBenchConfig {
    dataset: SweBenchDataset,
    dataset_id: String,
    split: String,
    num_samples: Option<usize>,
    token_env: Option<String>,
    timeout_secs: u64,
    host_repo_path: Option<PathBuf>,
    harness_image: String,
    build_images: bool,
    max_workers: usize,
    docker_socket_path: PathBuf,
    mount_docker_socket: bool,
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

#[derive(Debug, Serialize)]
struct SweBenchPrediction<'a> {
    instance_id: &'a str,
    model_name_or_path: &'a str,
    model_patch: &'a str,
}

impl super::Benchmark for SweBenchBenchmark {
    fn name(&self) -> &str {
        "swebench"
    }

    fn display_name(&self) -> &'static str {
        "SWE-Bench"
    }

    fn category(&self) -> crate::reports::model::BenchmarkCategory {
        crate::reports::model::BenchmarkCategory::LongContextCoding
    }

    fn to_report_result(&self, raw: &serde_json::Value) -> Result<BenchmarkResult> {
        use crate::reports::model::{Score, ScoreUnit};

        let resolution_rate = raw
            .get("resolution_rate")
            .and_then(|v| v.as_f64())
            .unwrap_or(0.0);
        let resolved = raw.get("resolved").and_then(|v| v.as_i64()).unwrap_or(0);
        let total_questions = raw
            .get("total_questions")
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
            "resolution_rate".to_string(),
            Score::float(resolution_rate, ScoreUnit::Percent)
                .primary(true)
                .higher_is_better(true),
        );
        scores.insert(
            "resolved".to_string(),
            Score::integer(resolved, ScoreUnit::Count),
        );
        scores.insert(
            "total_questions".to_string(),
            Score::integer(total_questions, ScoreUnit::Count),
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

        Ok(BenchmarkResult {
            scores,
            breakdowns: BTreeMap::new(),
            artifacts: vec![],
            diagnostics: vec![],
            raw: raw.clone(),
        })
    }

    fn pre_execute(&self, config: &yaml_serde::Value) -> Result<()> {
        let cfg = parse_config(SweBenchDataset::Basic, config)?;
        prepare_swebench(&cfg)?;
        Ok(())
    }

    fn execute(&self, model: &Model, config: &yaml_serde::Value) -> Result<serde_json::Value> {
        execute_swebench(SweBenchDataset::Basic, model, config)
    }
}

impl super::Benchmark for SweBenchVerifiedBenchmark {
    fn name(&self) -> &str {
        "swebench_verified"
    }

    fn display_name(&self) -> &'static str {
        "SWE-Bench Verified"
    }

    fn category(&self) -> crate::reports::model::BenchmarkCategory {
        crate::reports::model::BenchmarkCategory::LongContextCoding
    }

    fn to_report_result(&self, raw: &serde_json::Value) -> Result<BenchmarkResult> {
        use crate::reports::model::{Score, ScoreUnit};

        let resolution_rate = raw
            .get("resolution_rate")
            .and_then(|v| v.as_f64())
            .unwrap_or(0.0);
        let resolved = raw.get("resolved").and_then(|v| v.as_i64()).unwrap_or(0);
        let total_questions = raw
            .get("total_questions")
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
            "resolution_rate".to_string(),
            Score::float(resolution_rate, ScoreUnit::Percent)
                .primary(true)
                .higher_is_better(true),
        );
        scores.insert(
            "resolved".to_string(),
            Score::integer(resolved, ScoreUnit::Count),
        );
        scores.insert(
            "total_questions".to_string(),
            Score::integer(total_questions, ScoreUnit::Count),
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

        Ok(BenchmarkResult {
            scores,
            breakdowns: BTreeMap::new(),
            artifacts: vec![],
            diagnostics: vec![],
            raw: raw.clone(),
        })
    }

    fn pre_execute(&self, config: &yaml_serde::Value) -> Result<()> {
        let cfg = parse_config(SweBenchDataset::Verified, config)?;
        prepare_swebench(&cfg)?;
        Ok(())
    }

    fn execute(&self, model: &Model, config: &yaml_serde::Value) -> Result<serde_json::Value> {
        execute_swebench(SweBenchDataset::Verified, model, config)
    }
}

impl super::Benchmark for SweBenchProBenchmark {
    fn name(&self) -> &str {
        "swebench_pro"
    }

    fn display_name(&self) -> &'static str {
        "SWE-Bench Pro"
    }

    fn category(&self) -> crate::reports::model::BenchmarkCategory {
        crate::reports::model::BenchmarkCategory::LongContextCoding
    }

    fn to_report_result(&self, raw: &serde_json::Value) -> Result<BenchmarkResult> {
        use crate::reports::model::{BenchmarkResult, Score, ScoreUnit};

        let resolution_rate = raw
            .get("resolution_rate")
            .and_then(|v| v.as_f64())
            .unwrap_or(0.0);
        let resolved = raw.get("resolved").and_then(|v| v.as_i64()).unwrap_or(0);
        let total_questions = raw
            .get("total_questions")
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
            "resolution_rate".to_string(),
            Score::float(resolution_rate, ScoreUnit::Percent)
                .primary(true)
                .higher_is_better(true),
        );
        scores.insert(
            "resolved".to_string(),
            Score::integer(resolved, ScoreUnit::Count),
        );
        scores.insert(
            "total_questions".to_string(),
            Score::integer(total_questions, ScoreUnit::Count),
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

        Ok(BenchmarkResult {
            scores,
            breakdowns: BTreeMap::new(),
            artifacts: vec![],
            diagnostics: vec![],
            raw: raw.clone(),
        })
    }

    fn pre_execute(&self, config: &yaml_serde::Value) -> Result<()> {
        let cfg = parse_config(SweBenchDataset::Pro, config)?;
        prepare_swebench(&cfg)?;
        Ok(())
    }

    fn execute(&self, model: &Model, config: &yaml_serde::Value) -> Result<serde_json::Value> {
        execute_swebench(SweBenchDataset::Pro, model, config)
    }
}

fn execute_swebench(
    dataset: SweBenchDataset,
    model: &Model,
    config: &yaml_serde::Value,
) -> Result<serde_json::Value> {
    let cfg = parse_config(dataset, config)?;
    prepare_swebench(&cfg)?;
    let client = Client::new_with_model_params(&model.proxy, model.set_params.as_ref())?;
    let mut instances = load_or_download_dataset(&cfg)?;
    if let Some(limit) = cfg.num_samples {
        instances.truncate(limit);
    }

    let run_dir = Path::new("benchmark_results")
        .join("swe_bench_runs")
        .join(sanitize_path_component(&model.display_name))
        .join(cfg.dataset.benchmark_name());
    fs::create_dir_all(&run_dir)?;

    let predictions_path = run_dir.join("predictions.jsonl");
    let mut predictions_file = fs::File::create(&predictions_path)?;
    let mut total_output_tokens = 0u64;
    let mut total_thinking_tokens = 0u64;
    let mut instance_rows = Vec::new();

    for instance in &instances {
        let prompt = build_patch_prompt(instance);
        let (response, output_tokens, thinking_tokens) =
            client.chat_completion(&model.model_name, "", &prompt)?;
        let output_tokens = output_tokens.unwrap_or(0);
        let thinking_tokens = thinking_tokens.unwrap_or(0);
        total_output_tokens += output_tokens;
        total_thinking_tokens += thinking_tokens;
        let patch = extract_diff(&response);
        let prediction = SweBenchPrediction {
            instance_id: &instance.instance_id,
            model_name_or_path: &model.model_name,
            model_patch: &patch,
        };
        writeln!(predictions_file, "{}", serde_json::to_string(&prediction)?)?;
        instance_rows.push(serde_json::json!({
            "instance_id": instance.instance_id,
            "repo": instance.repo,
            "base_commit": instance.base_commit,
            "generated_patch": patch,
            "output_tokens": output_tokens,
            "thinking_tokens": thinking_tokens,
        }));
    }

    let eval_result = run_swebench_harness(&cfg, &run_dir, &predictions_path)?;
    let resolved = parse_resolved_count(&run_dir).unwrap_or(0);
    let total = instances.len();
    let resolution_rate = if total == 0 {
        0.0
    } else {
        resolved as f64 / total as f64
    };

    Ok(serde_json::json!({
        "dataset": cfg.dataset.benchmark_name(),
        "dataset_id": cfg.dataset_id,
        "split": cfg.split,
        "resolved": resolved,
        "total_questions": total,
        "resolution_rate": resolution_rate,
        "harness_passed": eval_result.passed,
        "timed_out": eval_result.timed_out,
        "exit_code": eval_result.exit_code,
        "error_summary": eval_result.error_summary,
        "stdout": truncate(&eval_result.stdout, 4000),
        "stderr": truncate(&eval_result.stderr, 4000),
        "predictions_path": predictions_path.display().to_string(),
        "output_tokens": total_output_tokens,
        "thinking_tokens": total_thinking_tokens,
        "instances": instance_rows,
    }))
}

#[derive(Debug, Clone)]
struct HarnessResult {
    passed: bool,
    timed_out: bool,
    exit_code: Option<i32>,
    stdout: String,
    stderr: String,
    error_summary: String,
}

fn prepare_swebench(cfg: &SweBenchConfig) -> Result<()> {
    ensure_swebench_harness_image(cfg)?;
    let _ = load_or_download_dataset(cfg)?;
    Ok(())
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
        if !cfg.docker_socket_path.exists() {
            return Err(anyhow::anyhow!(
                "SWE-Bench official harness requires Docker socket access, but {} does not exist. Set docker.mount_docker_socket=false only for harness images that do not need Docker, or configure docker.docker_socket_path.",
                cfg.docker_socket_path.display()
            ));
        }
        docker.mounts.push(DockerMount::direct_readwrite(
            &cfg.docker_socket_path,
            "/var/run/docker.sock",
        ));
    }
    docker.workdir = Some("/work".to_string());
    docker.host_repo_path = cfg.host_repo_path.clone();
    docker.name_prefix = "llm-benchmark-runner-swebench".to_string();
    // SWE-Bench harness needs network for repository/image/dataset setup unless all artifacts are pre-cached.
    docker.network_none = false;
    docker.read_only_root = false;
    docker.tmpfs.clear();
    docker.pids_limit = None;
    docker.memory = None;
    if let Some(token_env) = &cfg.token_env {
        if let Ok(token) = std::env::var(token_env) {
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
        .unwrap_or(true);
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

fn parse_resolved_count(run_dir: &Path) -> Option<usize> {
    let mut json_files = Vec::new();
    collect_json_files(run_dir, &mut json_files, 0).ok()?;
    json_files.sort();

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

fn truncate(value: &str, max_chars: usize) -> String {
    if value.chars().count() <= max_chars {
        return value.to_string();
    }
    let mut out: String = value.chars().take(max_chars).collect();
    out.push('…');
    out
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
}
