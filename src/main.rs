mod benchmarks;
mod client;
mod config;
mod report;
mod runner;

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

    if let Some(ref existing) = existing_results {
        if let Some(models) = existing.get("models").and_then(|v| v.as_object()) {
            for (name, data) in models {
                if let Some(completed) = data.get("benchmarks_completed").and_then(|v| v.as_array()) {
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

    // Model loop
    for model in &config.models {
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

        let (model_result, new_successful, new_failed) = runner::run_model(
            model,
            &benchmarks_to_run,
            &config.benchmark,
            &model_completed_benchmarks,
        )?;
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

    // Post-execute
    println!("\nPost-execution phase:");
    let mut kld_pairwise = serde_json::Map::new();
    if let Ok(kld_result) = benchmarks::post_execute_benchmark("kld", &all_models_results) {
        if let Some(map) = kld_result.as_object() {
            for (k, v) in map {
                kld_pairwise.insert(k.clone(), v.clone());
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
