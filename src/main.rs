use llm_benchmark_runner::benchmarks;
use llm_benchmark_runner::client;
use llm_benchmark_runner::config;
use llm_benchmark_runner::report;
use llm_benchmark_runner::runner;
use llm_benchmark_runner::utils;

mod mock_report;

use anyhow::Result;
use clap::{Parser, Subcommand};
use llm_benchmark_runner::reports::model::BenchmarkResult;
use std::collections::HashMap;
use std::fs;
use std::path::Path;

/// Signal handler for Ctrl-C (SIGINT): stops the current model process and exits.
fn stop_model_and_exit() {
    let mut pid_lock = runner::CURRENT_MODEL_PID
        .lock()
        .expect(llm_benchmark_runner::shared::MUTEX_PANIC_MSG);
    if let Some(pid) = *pid_lock {
        *pid_lock = None;
        drop(pid_lock); // Release the lock before sending signals

        // Send SIGTERM first, wait a second, then SIGKILL if still alive
        #[cfg(unix)]
        {
            use nix::sys::signal::{kill, Signal};
            use nix::unistd::Pid;
            let _ = kill(Pid::from_raw(pid as i32), Signal::SIGTERM);
            std::thread::sleep(std::time::Duration::from_secs(1));
            // Force kill the process group (in case it spawned children)
            let _ = kill(Pid::from_raw(-(pid as i32)), Signal::SIGKILL);
        }
        #[cfg(not(unix))]
        {
            unsafe {
                libc::kill(pid as libc::pid_t, libc::SIGTERM);
            }
            std::thread::sleep(std::time::Duration::from_secs(1));
            unsafe {
                libc::kill(-(pid as libc::pid_t), libc::SIGKILL);
            }
        }
    }
    // Exit the main process (guards will already be dropped if they're in scope,
    // but we don't rely on that; the model is already stopped)
    std::process::exit(1);
}

const DEFAULT_CONFIG: &str = "models_config.yaml";
const RESULTS_FILE: &str = "benchmark_results/results.json";

/// Per-model benchmark state recovered from a previously-saved results file,
/// used to resume an interrupted run.
struct ResumeState {
    completed_benchmarks_per_model: HashMap<String, Vec<String>>,
    failed_benchmarks_per_model: HashMap<String, Vec<String>>,
    all_models_results: HashMap<String, HashMap<String, BenchmarkResult>>,
}

impl ResumeState {
    fn empty() -> Self {
        Self {
            completed_benchmarks_per_model: HashMap::new(),
            failed_benchmarks_per_model: HashMap::new(),
            all_models_results: HashMap::new(),
        }
    }
}

/// Parse a previously-saved results JSON into per-model completed/failed lists and
/// full results, enabling resume of interrupted runs. Returns a `ResumeState`.
fn load_resume_state(existing_results: Option<&serde_json::Value>) -> ResumeState {
    let mut state = ResumeState::empty();

    if let Some(existing) = existing_results {
        if let Some(models) = existing.get("models").and_then(|v| v.as_object()) {
            for (name, data) in models {
                if let Some(completed) = data.get("benchmarks_completed").and_then(|v| v.as_array())
                {
                    state.completed_benchmarks_per_model.insert(
                        name.clone(),
                        completed
                            .iter()
                            .filter_map(|v| v.as_str().map(|s| s.to_string()))
                            .collect(),
                    );
                }
                if let Some(failed) = data.get("benchmarks_failed").and_then(|v| v.as_array()) {
                    state.failed_benchmarks_per_model.insert(
                        name.clone(),
                        failed
                            .iter()
                            .filter_map(|v| v.as_str().map(|s| s.to_string()))
                            .collect(),
                    );
                }
                // Deserialize the BenchmarkResult objects from the saved JSON.
                let mut per_model_benchmarks: HashMap<String, BenchmarkResult> = HashMap::new();
                if let Some(benchmark_data) = data.get("benchmarks").and_then(|v| v.as_object()) {
                    for (bench_name, bench_result) in benchmark_data {
                        if let Ok(result) =
                            serde_json::from_value::<BenchmarkResult>(bench_result.clone())
                        {
                            per_model_benchmarks.insert(bench_name.clone(), result);
                        }
                    }
                }
                state
                    .all_models_results
                    .insert(name.clone(), per_model_benchmarks);
            }
        }
    }

    state
}

/// Running-sum timing accumulator used for O(1) ETA estimation across the model
/// loop. Avoids rebuilding/scanning timing vectors each model iteration.
struct TimingAccumulator {
    global_sum: std::time::Duration,
    global_count: usize,
    bench_sum: HashMap<String, std::time::Duration>,
    bench_count: HashMap<String, usize>,
}

impl TimingAccumulator {
    fn new() -> Self {
        Self {
            global_sum: std::time::Duration::from_secs(0),
            global_count: 0,
            bench_sum: HashMap::new(),
            bench_count: HashMap::new(),
        }
    }

    fn merge(&mut self, per_bench_timings: HashMap<String, Vec<std::time::Duration>>) {
        for (bench_name, timings) in per_bench_timings {
            let s = self.bench_sum.entry(bench_name.clone()).or_default();
            for t in &timings {
                *s += *t;
            }
            *self.bench_count.entry(bench_name).or_default() += timings.len();
            self.global_sum += timings.iter().cloned().sum::<std::time::Duration>();
            self.global_count += timings.len();
        }
    }

    /// Estimate the total remaining runtime for future models/benchmarks, using
    /// the per-benchmark average where known, else the global overall average.
    fn estimate_remaining(
        &self,
        model_idx: usize,
        models: &[config::Model],
        benchmarks: &[String],
        completed_benchmarks_per_model: &HashMap<String, Vec<String>>,
    ) -> std::time::Duration {
        let overall_avg = if self.global_count == 0 {
            std::time::Duration::from_secs(0)
        } else {
            self.global_sum.div_f64(self.global_count as f64)
        };
        if model_idx + 1 >= models.len() {
            return std::time::Duration::from_secs(0);
        }
        let mut remaining_est: std::time::Duration = std::time::Duration::from_secs(0);
        for future_model in &models[model_idx + 1..] {
            let future_completed = completed_benchmarks_per_model
                .get(&future_model.display_name)
                .cloned()
                .unwrap_or_default();
            for bench in benchmarks {
                if !future_completed.contains(bench) {
                    let bench_avg = self.bench_sum.get(bench).map(|s| {
                        let n = *self.bench_count.get(bench).unwrap_or(&0);
                        if n > 0 {
                            s.div_f64(n as f64)
                        } else {
                            // No completed timings for this benchmark yet; fall
                            // back to the overall average to avoid NaN from
                            // div_f64(0.0).
                            overall_avg
                        }
                    });
                    remaining_est += bench_avg.unwrap_or(overall_avg);
                }
            }
        }
        remaining_est
    }
}

#[derive(Parser)]
#[command(name = "llm-benchmark-runner")]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    Run {
        #[arg(short, long, default_value = DEFAULT_CONFIG)]
        config: String,
        #[arg(long)]
        no_resume: bool,
    },
    TestModels {
        #[arg(short, long, default_value = DEFAULT_CONFIG)]
        config: String,
    },
    Report {
        #[arg(short, long, default_value = RESULTS_FILE)]
        results: String,
        #[arg(short, long, default_value = "benchmark_results")]
        output: String,
        #[arg(short = 'c', long, default_value = DEFAULT_CONFIG)]
        config: String,
    },
    /// Generate only comparison reports from existing results.
    Compare {
        #[arg(short = 'c', long, default_value = DEFAULT_CONFIG)]
        config: String,
        #[arg(short, long, default_value = RESULTS_FILE)]
        results: String,
        #[arg(short, long, default_value = "benchmark_results")]
        output: String,
    },
    /// Generate a mock report with synthetic minebench data for testing the voxel viewer.
    MockReport,
}

fn main() -> Result<()> {
    let cli = Cli::parse();

    // Install the Ctrl-C handler to stop the model process gracefully.
    ctrlc::set_handler(stop_model_and_exit).expect("Failed to set Ctrl-C handler");

    match cli.command {
        Commands::Run { config, no_resume } => run_benchmarks(&config, no_resume),
        Commands::TestModels { config } => test_models(&config),
        Commands::Report {
            results,
            output,
            config,
        } => generate_report(&results, &output, &config),
        Commands::MockReport => mock_report::generate(),
        Commands::Compare {
            config,
            results,
            output,
        } => generate_comparison_reports(&config, &results, &output),
    }
}

fn run_benchmarks(config_path: &str, no_resume: bool) -> Result<()> {
    println!("Loading config: {}", config_path);
    let config = config::load_config(config_path)?;
    if config.docker.mount_docker_socket {
        eprintln!(
            "⚠️  WARNING: mount_docker_socket is enabled — benchmark containers can access \
             the host Docker socket, granting full host root access on escape. Only use in \
             trusted environments."
        );
    }
    if config.models.is_empty() {
        return Err(anyhow::anyhow!("No models defined"));
    }
    let benchmarks: Vec<String> = if config.benchmarks.is_empty() {
        benchmarks::get_benchmark_names()
    } else {
        config.benchmarks.clone()
    };

    let existing_results = if !no_resume {
        load_existing_results(RESULTS_FILE)?
    } else {
        None
    };
    // Build completed and failed benchmark tracking per model from any saved run
    let s = load_resume_state(existing_results.as_ref());
    let (
        mut completed_benchmarks_per_model,
        mut failed_benchmarks_per_model,
        mut all_models_results,
    ) = (
        s.completed_benchmarks_per_model,
        s.failed_benchmarks_per_model,
        s.all_models_results,
    );

    // Running-sum accumulator for O(1) ETA each model iteration.
    let mut timings = TimingAccumulator::new();

    println!("Benchmarks: {}", benchmarks.join(", "));

    // Pre-execute
    for bench_name in &benchmarks {
        let bench_cfg = config::attach_docker_config(
            config
                .benchmark
                .get(bench_name)
                .cloned()
                .unwrap_or(yaml_serde::Value::Null),
            &config.docker,
        );
        if let Err(e) = benchmarks::pre_execute_benchmark(bench_name, &bench_cfg) {
            eprintln!("Warning: pre-execute {} failed: {}", bench_name, e);
        }
    }

    let total_models = config.models.len();
    let run_start = std::time::Instant::now();

    // Model loop
    for (model_idx, model) in config.models.iter().enumerate() {
        let model_completed_benchmarks = completed_benchmarks_per_model
            .get(&model.display_name)
            .cloned()
            .unwrap_or_default();
        let model_failed_benchmarks = failed_benchmarks_per_model
            .get(&model.display_name)
            .cloned()
            .unwrap_or_default();

        // Check if all benchmarks are completed (no failed ones to retry)
        let successful_count = model_completed_benchmarks.len();
        let failed_count = model_failed_benchmarks.len();
        if successful_count + failed_count == benchmarks.len() && failed_count == 0 {
            println!("\nSkipping completed model: {}", model.display_name);
            continue;
        }

        // Only re-run the failed benchmarks; use completed list for context
        let benchmarks_to_run: Vec<String> = if !model_failed_benchmarks.is_empty() {
            model_failed_benchmarks.clone()
        } else {
            benchmarks.clone()
        };

        let (model_result, new_successful, new_failed, per_bench_timings) = runner::run_model(
            model,
            &benchmarks_to_run,
            &config.benchmark,
            &config.docker,
            &model_completed_benchmarks,
        )?;
        // Merge per-model timings into running-sum accumulator.
        timings.merge(per_bench_timings);

        // Compute ETA: sum of estimated times for all remaining (model, benchmark) pairs.
        let remaining_est = timings.estimate_remaining(
            model_idx,
            &config.models,
            &benchmarks,
            &completed_benchmarks_per_model,
        );
        let eta_str = if remaining_est.is_zero() {
            "–".to_string()
        } else {
            utils::format_duration(remaining_est)
        };
        let total_runtime = run_start.elapsed();
        let runtime_str = utils::format_duration(total_runtime);
        println!(
            "\n  [model {}/{}] {} runtime: {}, ETA remaining: {}",
            model_idx + 1,
            total_models,
            model.display_name,
            runtime_str,
            eta_str
        );

        // Update in-memory results: merge the re-run (previously failed) results
        // into any already-completed results for this model. Replacing the whole
        // entry would discard successful benchmarks completed before the resume.
        all_models_results
            .entry(model.display_name.clone())
            .or_default()
            .extend(model_result);

        // Merge newly-completed benchmarks into the model's completed list so
        // benchmarks finished in an earlier run are not forgotten across resumes
        // (otherwise the next resume would re-run them).
        let mut merged_completed = model_completed_benchmarks.clone();
        for b in new_successful {
            if !merged_completed.contains(&b) {
                merged_completed.push(b);
            }
        }
        completed_benchmarks_per_model.insert(model.display_name.clone(), merged_completed);
        failed_benchmarks_per_model.insert(model.display_name.clone(), new_failed);

        save_results(
            &all_models_results,
            &completed_benchmarks_per_model,
            &failed_benchmarks_per_model,
            RESULTS_FILE,
        )?;
    }

    // Total runtime for the entire run
    let total_runtime = run_start.elapsed();
    let runtime_str = utils::format_duration(total_runtime);
    println!("\nTotal runtime: {}", runtime_str);

    // Post-execute for each benchmark (collects KLD pairwise + post results).
    let (kld_pairwise, post_execute_results) = run_post_execute(&benchmarks, &all_models_results);

    // Save final results JSON (with both models and kld_pairwise).
    let output_dir = Path::new("benchmark_results");
    save_final_results(&all_models_results, &kld_pairwise)?;

    // Generate reports from in-memory results, passing per-benchmark, per-model BenchmarkResult objects
    report::generate_reports(
        &all_models_results,
        output_dir,
        &config.comparisons,
        &post_execute_results,
    )?;
    println!("\nBenchmark complete.");
    Ok(())
}

/// Run the post-execution phase for every benchmark, aggregating per-model results.
/// Returns the KLD pairwise map and the post-execute results.
fn run_post_execute(
    benchmarks: &[String],
    all_models_results: &HashMap<String, HashMap<String, BenchmarkResult>>,
) -> (
    serde_json::Map<String, serde_json::Value>,
    HashMap<String, BenchmarkResult>,
) {
    println!("\nPost-execution phase:");
    let mut kld_pairwise: serde_json::Map<String, serde_json::Value> = serde_json::Map::new();
    let mut post_execute_results: HashMap<String, BenchmarkResult> = HashMap::new();
    for bench_name in benchmarks {
        // Collect results from all models for this benchmark
        let model_results: HashMap<String, BenchmarkResult> = all_models_results
            .iter()
            .filter_map(|(model_name, bench_results)| {
                bench_results
                    .get(bench_name)
                    .map(|result| (model_name.clone(), result.clone()))
            })
            .collect();

        if model_results.is_empty() {
            eprintln!(
                "Warning: No model results for benchmark {}, skipping post_execute",
                bench_name
            );
            continue;
        }

        match benchmarks::post_execute_benchmark(bench_name, &model_results) {
            Ok(post_result) => {
                post_execute_results.insert(bench_name.clone(), post_result.clone());
                if bench_name == "kld" {
                    // Extract raw JSON from the KLD post-execute result
                    if let Ok(value) = serde_json::to_value(&post_result) {
                        if let Some(map) = value.as_object() {
                            for (k, v) in map {
                                kld_pairwise.insert(k.clone(), v.clone());
                            }
                        }
                    }
                }
            }
            Err(e) => {
                eprintln!("Warning: post-execute {} failed: {}", bench_name, e);
            }
        }
    }
    (kld_pairwise, post_execute_results)
}

/// Write the final results JSON (models + kld_pairwise) atomically to RESULTS_FILE.
fn save_final_results(
    all_models_results: &HashMap<String, HashMap<String, BenchmarkResult>>,
    kld_pairwise: &serde_json::Map<String, serde_json::Value>,
) -> Result<()> {
    fs::create_dir_all(Path::new("benchmark_results"))?;

    // Build the final JSON for saving: models with all benchmark results + kld_pairwise
    let mut models_json = serde_json::Map::new();
    for (model_name, bench_results) in all_models_results {
        let mut model_data = serde_json::Map::new();
        let mut bench_json = serde_json::Map::new();
        for (bench_name, result) in bench_results {
            bench_json.insert(
                bench_name.clone(),
                serde_json::to_value(result).unwrap_or(serde_json::json!(null)),
            );
        }
        model_data.insert(
            "benchmarks".to_string(),
            serde_json::Value::Object(bench_json),
        );
        models_json.insert(model_name.clone(), serde_json::Value::Object(model_data));
    }
    let final_results = serde_json::json!({
        "models": models_json,
        "kld_pairwise": kld_pairwise,
    });
    let tmp_path = format!("{}.tmp", RESULTS_FILE);
    let json = serde_json::to_string_pretty(&final_results)?;
    fs::write(&tmp_path, json)?;
    fs::rename(&tmp_path, RESULTS_FILE)?;
    Ok(())
}

fn test_models(config_path: &str) -> Result<()> {
    // NOTE: Early termination (Ctrl+C) may leave models running.
    // To handle this robustly, integrate a signal handler (e.g., ctrlc) to
    // stop all running processes. This is consistent with the existing
    // `run_model` behavior. Future improvement: add SIGINT handling here.

    println!("Loading config: {}", config_path);
    let config = config::load_config(config_path)?;
    if config.models.is_empty() {
        return Err(anyhow::anyhow!("No models defined in config"));
    }

    let mut results: Vec<(String, bool, String)> = Vec::new(); // name, success, message

    for model in &config.models {
        println!("\n  Testing model: {} ...", model.display_name);
        let process = match runner::start_model(&model.cmd) {
            Ok(p) => p,
            Err(e) => {
                eprintln!("  FAIL: Failed to start model: {}", e);
                results.push((
                    model.display_name.clone(),
                    false,
                    format!("start error: {}", e),
                ));
                continue;
            }
        };

        let mut process_guard = runner::ModelProcessGuard::new(process, model.cmd_stop.clone());

        let client = match client::Client::new(&model.proxy) {
            Ok(c) => c,
            Err(e) => {
                eprintln!("  FAIL: Failed to create client: {}", e);
                results.push((
                    model.display_name.clone(),
                    false,
                    format!("client error: {}", e),
                ));
                continue;
            }
        };

        if !runner::wait_for_health(&client) {
            eprintln!("  FAIL: Model proxy did not become healthy");
            results.push((
                model.display_name.clone(),
                false,
                "proxy not healthy".to_string(),
            ));
            continue;
        }
        println!("  Proxy healthy.");

        // Send test prompt
        let test_prompt = "Say hello in one word.";
        let test_system = "You are a helpful assistant.";
        let model_name_for_api = &model.model_name;
        match client.chat_completion(model_name_for_api, test_system, test_prompt) {
            Ok((response, output_tokens, thinking_tokens)) => {
                println!("  Prompt: {}", test_prompt);
                println!("  Response: {}", response);
                println!(
                    "  Tokens: output={}, thinking={}",
                    output_tokens
                        .map(|v| v.to_string())
                        .unwrap_or_else(|| "–".to_string()),
                    thinking_tokens
                        .map(|v| v.to_string())
                        .unwrap_or_else(|| "–".to_string())
                );
                println!("  SUCCESS");
                results.push((model.display_name.clone(), true, response.clone()));
            }
            Err(e) => {
                eprintln!("  FAIL: Chat completion failed: {}", e);
                results.push((
                    model.display_name.clone(),
                    false,
                    format!("chat error: {}", e),
                ));
            }
        }

        println!("  Stopping model: {}", model.display_name);
        process_guard.stop();
    }

    // Print summary
    println!("\n=== Test Summary ===");
    for (name, success, msg) in &results {
        if *success {
            // Truncate the response snippet to keep the summary readable
            let truncated_msg = if msg.len() > 50 {
                format!("{}...", &msg[..47])
            } else {
                msg.clone()
            };
            println!("  [PASS] {} - {}", name, truncated_msg);
        } else {
            println!("  [FAIL] {} - {}", name, msg);
        }
    }
    let pass_count = results.iter().filter(|(_, s, _)| *s).count();
    let fail_count = results.iter().filter(|(_, s, _)| !*s).count();
    println!(
        "  Total: {} tests, {} passed, {} failed",
        results.len(),
        pass_count,
        fail_count
    );

    if fail_count > 0 {
        Err(anyhow::anyhow!("Some model tests failed"))
    } else {
        Ok(())
    }
}

fn load_existing_results(path: &str) -> Result<Option<serde_json::Value>> {
    if !Path::new(path).exists() {
        return Ok(None);
    }
    let content = fs::read_to_string(path)?;
    let results: serde_json::Value = serde_json::from_str(&content)?;
    Ok(Some(results))
}

fn save_results(
    all_models_results: &HashMap<String, HashMap<String, BenchmarkResult>>,
    completed_per_model: &HashMap<String, Vec<String>>,
    failed_per_model: &HashMap<String, Vec<String>>,
    path: &str,
) -> Result<()> {
    // Convert all_models_results to serde_json::Value, attaching each model's
    // OWN completed/failed lists (not the most recently processed model's).
    let models_json: serde_json::Map<String, serde_json::Value> = all_models_results
        .iter()
        .map(|(model_name, bench_results)| {
            let bench_values: serde_json::Map<String, serde_json::Value> = bench_results
                .iter()
                .map(|(bench_name, result)| {
                    (
                        bench_name.clone(),
                        serde_json::to_value(result).unwrap_or(serde_json::json!(null)),
                    )
                })
                .collect();
            let model_value = serde_json::json!({
                "benchmarks": bench_values,
                "benchmarks_completed": completed_per_model
                    .get(model_name)
                    .cloned()
                    .unwrap_or_default(),
                "benchmarks_failed": failed_per_model
                    .get(model_name)
                    .cloned()
                    .unwrap_or_default(),
            });
            (model_name.clone(), model_value)
        })
        .collect();

    let result = serde_json::json!({ "models": models_json });
    let tmp_path = format!("{}.tmp", path);
    let json = serde_json::to_string_pretty(&result)?;
    fs::write(&tmp_path, json)?;
    fs::rename(&tmp_path, path)?;
    Ok(())
}

fn generate_report(results_path: &str, output_dir: &str, config_path: &str) -> Result<()> {
    // Load JSON and convert to in-memory BenchmarkResult objects
    let content = fs::read_to_string(results_path)?;
    let results_json: serde_json::Value = serde_json::from_str(&content)?;
    let output_path = Path::new(output_dir);
    fs::create_dir_all(output_path)?;
    // Load config for comparisons
    let comparisons = if let Ok(config) = config::load_config(config_path) {
        config.comparisons
    } else {
        Vec::new()
    };
    // Deserialize the models and benchmarks into BenchmarkResult objects
    let all_models_results: HashMap<String, HashMap<String, BenchmarkResult>> = results_json
        .get("models")
        .and_then(|v| v.as_object())
        .map(|models_obj| {
            models_obj
                .iter()
                .filter_map(|(model_name, model_data)| {
                    model_data
                        .get("benchmarks")
                        .and_then(|v| v.as_object())
                        .map(|benchmarks_obj| {
                            let bench_results: HashMap<String, BenchmarkResult> = benchmarks_obj
                                .iter()
                                .filter_map(|(bench_name, bench_result)| {
                                    serde_json::from_value(bench_result.clone())
                                        .ok()
                                        .map(|r| (bench_name.clone(), r))
                                })
                                .collect();
                            (model_name.clone(), bench_results)
                        })
                })
                .collect()
        })
        .unwrap_or_default();

    // No post-execute aggregate when running report from JSON; just pass empty post_execute_results
    let post_execute_results: HashMap<String, BenchmarkResult> = HashMap::new();
    report::generate_reports(
        &all_models_results,
        output_path,
        &comparisons,
        &post_execute_results,
    )?;
    Ok(())
}

/// Generate only comparison reports from existing results (standalone `compare` command).
fn generate_comparison_reports(
    config_path: &str,
    results_path: &str,
    output_dir: &str,
) -> Result<()> {
    let config = config::load_config(config_path)?;
    if config.comparisons.is_empty() {
        println!("No comparisons defined in config.");
        return Ok(());
    }

    // Load JSON and deserialize to in-memory BenchmarkResult objects
    let content = fs::read_to_string(results_path)?;
    let results_json: serde_json::Value = serde_json::from_str(&content)?;
    let output_path = Path::new(output_dir);
    fs::create_dir_all(output_path)?;

    let all_models_results: HashMap<String, HashMap<String, BenchmarkResult>> = results_json
        .get("models")
        .and_then(|v| v.as_object())
        .map(|models_obj| {
            models_obj
                .iter()
                .filter_map(|(model_name, model_data)| {
                    model_data
                        .get("benchmarks")
                        .and_then(|v| v.as_object())
                        .map(|benchmarks_obj| {
                            let bench_results: HashMap<String, BenchmarkResult> = benchmarks_obj
                                .iter()
                                .filter_map(|(bench_name, bench_result)| {
                                    serde_json::from_value(bench_result.clone())
                                        .ok()
                                        .map(|r| (bench_name.clone(), r))
                                })
                                .collect();
                            (model_name.clone(), bench_results)
                        })
                })
                .collect()
        })
        .unwrap_or_default();

    // No post-execute aggregate when running from JSON; pass empty post_execute_results
    let post_execute_results: HashMap<String, BenchmarkResult> = HashMap::new();

    // Generate comparison reports using the in-memory results
    for (idx, comparison) in config.comparisons.iter().enumerate() {
        if comparison.models.is_empty() {
            continue;
        }
        let slug = utils::slugify(&comparison.title);
        let filename = if slug.is_empty() {
            format!("comparison-{}.html", idx)
        } else {
            format!("comparison-{}.html", slug)
        };

        report::generate_comparison_report(
            &all_models_results,
            output_path,
            &filename,
            comparison,
            &post_execute_results,
        )?;
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn load_resume_state_parses_saved_results() {
        let json = serde_json::json!({
            "models": {
                "MyModel Q4": {
                    "status": "completed",
                    "benchmarks_completed": ["mmlu_pro"],
                    "benchmarks": {
                        "mmlu_pro": {
                            "scores": {},
                            "breakdowns": {},
                            "error_classification": {},
                            "artifacts": [],
                            "diagnostics": [],
                            "raw": {}
                        }
                    }
                }
            }
        });
        let st = load_resume_state(Some(&json));
        assert_eq!(
            st.completed_benchmarks_per_model.get("MyModel Q4").unwrap(),
            &vec!["mmlu_pro".to_string()]
        );
        assert!(st.failed_benchmarks_per_model.is_empty());
        let model = st.all_models_results.get("MyModel Q4").unwrap();
        assert!(model.contains_key("mmlu_pro"));
    }

    #[test]
    fn load_resume_state_none_is_empty() {
        let st = load_resume_state(None);
        assert!(st.completed_benchmarks_per_model.is_empty());
        assert!(st.failed_benchmarks_per_model.is_empty());
        assert!(st.all_models_results.is_empty());
    }

    #[test]
    fn timing_accumulator_advances_and_estimates() {
        let mut t = TimingAccumulator::new();
        let mut per_bench = HashMap::new();
        per_bench.insert(
            "mmlu_pro".to_string(),
            vec![
                std::time::Duration::from_secs(4),
                std::time::Duration::from_secs(6),
            ],
        );
        t.merge(per_bench);

        // No future models → remaining estimate is zero.
        assert_eq!(
            t.estimate_remaining(0, &[], &["mmlu_pro".to_string()], &HashMap::new()),
            std::time::Duration::from_secs(0)
        );

        // One future model that still needs "mmlu_pro" → estimated via bench avg (5s).
        let completed = HashMap::new();
        let make_model = |name: &str| config::Model {
            model_name: name.into(),
            display_name: name.into(),
            cmd: "".into(),
            proxy: "".into(),
            cmd_stop: None,
            set_params: None,
            rate_limit_ms: None,
        };
        let models = vec![make_model("M1"), make_model("M2")];
        // model_idx=0 with 2 models → one future model remains.
        let est = t.estimate_remaining(0, &models, &["mmlu_pro".to_string()], &completed);
        // Average of 4s and 6s = 5s.
        assert_eq!(est, std::time::Duration::from_secs(5));
    }
}
