mod benchmarks;
mod client;
mod config;
mod report;
mod runner;
mod utils;

use anyhow::Result;
use clap::{Parser, Subcommand};
use std::collections::HashMap;
use std::fs;
use std::path::Path;

const DEFAULT_CONFIG: &str = "models_config.yaml";
const RESULTS_FILE: &str = "benchmark_results/results.json";

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
    },
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    match cli.command {
        Commands::Run { config, no_resume } => run_benchmarks(&config, no_resume),
        Commands::TestModels { config } => test_models(&config),
        Commands::Report { results, output } => generate_report(&results, &output),
    }
}

fn run_benchmarks(config_path: &str, no_resume: bool) -> Result<()> {
    println!("Loading config: {}", config_path);
    let config = config::load_config(config_path)?;
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
    // Build completed and failed benchmark tracking per model
    let mut completed_benchmarks_per_model: HashMap<String, Vec<String>> = HashMap::new();
    let mut failed_benchmarks_per_model: HashMap<String, Vec<String>> = HashMap::new();
    let mut all_models_results: HashMap<String, serde_json::Value> = HashMap::new();

    // Global timing map: benchmark name -> durations (across all models)
    let mut global_timings: HashMap<String, Vec<std::time::Duration>> = HashMap::new();

    if let Some(ref existing) = existing_results {
        if let Some(models) = existing.get("models").and_then(|v| v.as_object()) {
            for (name, data) in models {
                if let Some(completed) = data.get("benchmarks_completed").and_then(|v| v.as_array())
                {
                    let bench_names: Vec<String> = completed
                        .iter()
                        .filter_map(|v| v.as_str().map(|s| s.to_string()))
                        .collect();
                    completed_benchmarks_per_model.insert(name.clone(), bench_names);
                }
                if let Some(failed) = data.get("benchmarks_failed").and_then(|v| v.as_array()) {
                    let bench_names: Vec<String> = failed
                        .iter()
                        .filter_map(|v| v.as_str().map(|s| s.to_string()))
                        .collect();
                    failed_benchmarks_per_model.insert(name.clone(), bench_names);
                }
                all_models_results.insert(name.clone(), data.clone());
            }
        }
    }

    println!("Benchmarks: {}", benchmarks.join(", "));

    // Pre-execute
    for bench_name in &benchmarks {
        let bench_cfg = config
            .benchmark
            .get(bench_name)
            .cloned()
            .unwrap_or(serde_yaml::Value::Null);
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
            &model_completed_benchmarks,
        )?;
        // Merge per-model timings into global map
        for (bench_name, timings) in per_bench_timings {
            global_timings
                .entry(bench_name)
                .or_default()
                .extend(timings);
        }

        // Compute ETA: sum of estimated times for all remaining (model, benchmark) pairs
        // using global average per benchmark (or global overall average)
        let mut remaining_est: std::time::Duration = std::time::Duration::from_secs(0);
        let all_durations: Vec<std::time::Duration> =
            global_timings.values().flatten().cloned().collect();
        let overall_avg = if all_durations.is_empty() {
            std::time::Duration::from_secs(0)
        } else {
            let sum: std::time::Duration = all_durations.iter().cloned().sum();
            sum.div_f64(all_durations.len() as f64)
        };

        // Count remaining (model, benchmark) pairs for future models
        for future_model in &config.models[model_idx + 1..] {
            let future_completed = completed_benchmarks_per_model
                .get(&future_model.display_name)
                .cloned()
                .unwrap_or_default();
            for bench in &benchmarks {
                if !future_completed.contains(bench) {
                    let bench_avg = global_timings.get(bench).map(|v| {
                        let sum: std::time::Duration = v.iter().cloned().sum();
                        sum.div_f64(v.len() as f64)
                    });
                    let est = bench_avg.unwrap_or(overall_avg);
                    remaining_est += est;
                }
            }
        }
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

        let status = if model_result.as_object().is_some_and(|m| !m.is_empty()) {
            "completed"
        } else {
            "error"
        };
        let mut model_data = serde_json::Map::new();
        model_data.insert("status".to_string(), serde_json::json!(status));
        model_data.insert(
            "benchmarks_completed".to_string(),
            serde_json::json!(new_successful.clone()),
        );
        model_data.insert(
            "benchmarks_failed".to_string(),
            serde_json::json!(new_failed.clone()),
        );
        if let Some(obj) = model_result.as_object() {
            for (k, v) in obj {
                model_data.insert(k.clone(), v.clone());
            }
        }
        all_models_results.insert(
            model.display_name.clone(),
            serde_json::Value::Object(model_data),
        );
        save_results(&all_models_results, &serde_json::Map::new(), RESULTS_FILE)?;
    }

    // Total runtime for the entire run
    let total_runtime = run_start.elapsed();
    let runtime_str = utils::format_duration(total_runtime);
    println!("\nTotal runtime: {}", runtime_str);

    // Post-execute
    println!("\nPost-execution phase:");
    let mut kld_pairwise = serde_json::Map::new();
    for bench_name in &benchmarks {
        match benchmarks::post_execute_benchmark(bench_name, &all_models_results) {
            Ok(post_result) => {
                if bench_name == "kld" {
                    if let Some(map) = post_result.as_object() {
                        for (k, v) in map {
                            kld_pairwise.insert(k.clone(), v.clone());
                        }
                    }
                }
            }
            Err(e) => {
                eprintln!("Warning: post-execute {} failed: {}", bench_name, e);
            }
        }
    }
    let final_results = serde_json::json!({
        "models": all_models_results,
        "kld_pairwise": kld_pairwise,
    });
    save_results(&all_models_results, &kld_pairwise, RESULTS_FILE)?;

    let output_dir = Path::new("benchmark_results");
    fs::create_dir_all(output_dir)?;
    report::generate_reports(&final_results, output_dir)?;
    println!("\nBenchmark complete.");
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

        let client = match client::Client::new(&model.proxy) {
            Ok(c) => c,
            Err(e) => {
                runner::stop_model(&model.cmd_stop, process);
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
            runner::stop_model(&model.cmd_stop, process);
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
        runner::stop_model(&model.cmd_stop, process);
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
    models: &HashMap<String, serde_json::Value>,
    kld_pairwise: &serde_json::Map<String, serde_json::Value>,
    path: &str,
) -> Result<()> {
    let result = serde_json::json!({ "models": models, "kld_pairwise": kld_pairwise });
    let tmp_path = format!("{}.tmp", path);
    let json = serde_json::to_string_pretty(&result)?;
    fs::write(&tmp_path, json)?;
    fs::rename(&tmp_path, path)?;
    Ok(())
}

fn generate_report(results_path: &str, output_dir: &str) -> Result<()> {
    let content = fs::read_to_string(results_path)?;
    let results: serde_json::Value = serde_json::from_str(&content)?;
    let output_path = Path::new(output_dir);
    fs::create_dir_all(output_path)?;
    report::generate_reports(&results, output_path)?;
    Ok(())
}
