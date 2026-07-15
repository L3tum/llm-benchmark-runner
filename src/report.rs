use crate::benchmarks::aime::AimeBenchmark;
use crate::benchmarks::coding_eval::CodingEvalBenchmark;
use crate::benchmarks::gpqa::GpqaBenchmark;
use crate::benchmarks::kld::KldBenchmark;
use crate::benchmarks::math500::Math500Benchmark;
use crate::benchmarks::minebench::MinebenchBenchmark;
use crate::benchmarks::mmlu_pro::MmluProBenchmark;
use crate::benchmarks::swe_bench::{
    SweBenchBenchmark, SweBenchProBenchmark, SweBenchVerifiedBenchmark,
};
use crate::config::Comparison;
use crate::reports::console::ConsoleReportGenerator;
use crate::reports::generator::{ReportContext, ReportGenerator};
use crate::reports::html::HtmlReportGenerator;
use crate::reports::markdown::MarkdownReportGenerator;
use crate::reports::model::{BenchmarkResult, ReportInput, ScoreValue, TestName, TestReportData};
use anyhow::Result;
use std::collections::{BTreeMap, HashMap};
use std::fs;
use std::path::Path;
use std::sync::OnceLock;

/// Ordered list of category names for display (includes reserved empty categories).
static CATEGORY_ORDER: OnceLock<Vec<String>> = OnceLock::new();

fn get_category_order() -> &'static Vec<String> {
    CATEGORY_ORDER.get_or_init(|| {
        vec![
            "Knowledge".to_string(),
            "Math".to_string(),
            "Short-Context-Coding".to_string(),
            "Long-Context-Coding".to_string(),
            "Creative".to_string(),
            "Reasoning".to_string(),
            "Research".to_string(),
            "Similarity".to_string(),
        ]
    })
}

fn slugify_name(name: String) -> String {
    name.to_lowercase()
        .replace(" ", "-")
        .replace("/", "-")
        .replace("_", "-")
        .replace("  ", "-")
        .trim()
        .to_string()
}

/// Build a `ReportInput` from in-memory `BenchmarkResult` objects.
fn build_report_input(
    all_models_results: &HashMap<String, HashMap<String, BenchmarkResult>>,
    post_execute_results: &HashMap<String, BenchmarkResult>,
) -> ReportInput {
    let models_evaluated: Vec<String> = all_models_results.keys().cloned().collect();
    let timestamp = chrono::Utc::now()
        .format("%Y-%m-%d %H:%M:%S UTC")
        .to_string();

    let benchmarks: Vec<&dyn crate::benchmarks::Benchmark> = vec![
        &MmluProBenchmark,
        &GpqaBenchmark,
        &AimeBenchmark,
        &Math500Benchmark,
        &MinebenchBenchmark,
        &CodingEvalBenchmark,
        &SweBenchBenchmark,
        &SweBenchVerifiedBenchmark,
        &SweBenchProBenchmark,
        &KldBenchmark,
    ];

    let mut tests = BTreeMap::new();

    for benchmark in benchmarks {
        // Collect per-model BenchmarkResult for this benchmark
        let mut model_results: BTreeMap<String, BenchmarkResult> = BTreeMap::new();

        for (model_name, bench_results) in all_models_results {
            if let Some(bench_result) = bench_results.get(benchmark.name()) {
                // Call to_report_result on the in-memory result
                match benchmark.to_report_result(bench_result) {
                    Ok(result) => {
                        model_results.insert(model_name.clone(), result);
                    }
                    Err(e) => {
                        eprintln!(
                            "Warning: Failed to convert {} result for {}: {}",
                            benchmark.name(),
                            model_name,
                            e
                        );
                    }
                }
            }
        }

        // Aggregate results (from post_execute)
        let aggregate = post_execute_results
            .get(benchmark.name())
            .and_then(|post_result| benchmark.to_report_aggregate(post_result).ok().flatten());

        if !model_results.is_empty() || aggregate.is_some() {
            tests.insert(
                TestName::new(benchmark.name()),
                TestReportData {
                    name: TestName::new(benchmark.name()),
                    display_name: benchmark.display_name().to_string(),
                    category: benchmark.category(),
                    model_results,
                    aggregate,
                },
            );
        }
    }

    let summary = generate_summary_from_tests(&tests);

    // Build the raw results JSON for backwards compatibility
    let raw_results = build_raw_results_json(all_models_results);

    ReportInput {
        generated_at: timestamp,
        models: models_evaluated,
        tests,
        summary,
        raw_results,
    }
}

/// Build raw JSON from in-memory results for backwards compatibility and saving.
fn build_raw_results_json(
    all_models_results: &HashMap<String, HashMap<String, BenchmarkResult>>,
) -> serde_json::Value {
    let mut models = serde_json::Map::new();
    for (model_name, bench_results) in all_models_results {
        let mut bench_json = serde_json::Map::new();
        for (bench_name, result) in bench_results {
            bench_json.insert(bench_name.clone(), serde_json::to_value(result).unwrap());
        }
        models.insert(
            model_name.clone(),
            serde_json::json!({ "benchmarks": bench_json }),
        );
    }
    serde_json::json!({ "models": models })
}

fn parse_optional_tokens(s: &str) -> Option<i64> {
    if s.is_empty() || s == "–" {
        None
    } else {
        s.parse::<i64>().ok()
    }
}

/// Generate a summary list from the normalized test data.
/// Finds the best model per benchmark using primary scores.
fn generate_summary_from_tests(tests: &BTreeMap<TestName, TestReportData>) -> Vec<String> {
    let mut summary = Vec::new();

    for test_data in tests.values() {
        if test_data.model_results.is_empty() {
            continue;
        }

        // Find the model with the best primary score
        let primary_scores: Vec<(&String, &BenchmarkResult, String, ScoreValue, Option<bool>)> =
            test_data
                .model_results
                .iter()
                .filter_map(|(model, result)| {
                    // Find the primary score
                    result.scores.iter().find_map(|(score_name, score)| {
                        if score.primary {
                            Some((
                                model,
                                result,
                                score_name.clone(),
                                score.value.clone(),
                                score.higher_is_better,
                            ))
                        } else {
                            None
                        }
                    })
                })
                .collect();

        if primary_scores.is_empty() {
            continue;
        }

        // Respect higher_is_better flag: default to true if unspecified
        let higher_is_better = primary_scores
            .first()
            .map(|s| s.4.unwrap_or(true))
            .unwrap_or(true);

        let best = if higher_is_better {
            primary_scores
                .iter()
                .cloned()
                .max_by(|(_, _, _, a, _), (_, _, _, b, _)| match (a, b) {
                    (ScoreValue::Float(f1), ScoreValue::Float(f2)) => {
                        f1.partial_cmp(f2).unwrap_or(std::cmp::Ordering::Equal)
                    }
                    (ScoreValue::Integer(i1), ScoreValue::Integer(i2)) => i1.cmp(i2),
                    (ScoreValue::Bool(b1), ScoreValue::Bool(b2)) => b1.cmp(b2),
                    (ScoreValue::Text(t1), ScoreValue::Text(t2)) => t1.cmp(t2),
                    _ => std::cmp::Ordering::Equal,
                })
        } else {
            primary_scores
                .iter()
                .cloned()
                .min_by(|(_, _, _, a, _), (_, _, _, b, _)| match (a, b) {
                    (ScoreValue::Float(f1), ScoreValue::Float(f2)) => {
                        f1.partial_cmp(f2).unwrap_or(std::cmp::Ordering::Equal)
                    }
                    (ScoreValue::Integer(i1), ScoreValue::Integer(i2)) => i1.cmp(i2),
                    (ScoreValue::Bool(b1), ScoreValue::Bool(b2)) => b1.cmp(b2),
                    (ScoreValue::Text(t1), ScoreValue::Text(t2)) => t1.cmp(t2),
                    _ => std::cmp::Ordering::Equal,
                })
        };

        if let Some((model, _, score_name, score_value, _)) = best {
            let formatted_score = match score_value {
                ScoreValue::Float(f) => format!("{:.1}%", f * 100.0),
                ScoreValue::Integer(i) => format!("{}", i),
                ScoreValue::Bool(b) => (if b { "✓" } else { "✗" }).to_string(),
                ScoreValue::Text(t) => t.to_string(),
                ScoreValue::Missing => String::from("–"),
            };

            let metric = score_name.as_str();
            let _display_name = test_data.display_name.as_str();
            summary.push(format!("Best {}: {} ({})", metric, model, formatted_score));
        }

        // KLD pairwise aggregate info
        if let Some(ref agg) = test_data.aggregate {
            if let Some(table) = agg.breakdowns.get("pairwise_kld") {
                for (pair, rows) in table.rows.iter() {
                    if let Some(kld_score) = rows.get("avg_kld") {
                        let formatted_kld = match &kld_score.value {
                            ScoreValue::Float(f) => format!("{:.3}", f),
                            _ => "N/A".to_string(),
                        };
                        if let Some(num_prompts) = rows.get("num_prompts_evaluated") {
                            let formatted_prompts = match &num_prompts.value {
                                ScoreValue::Integer(i) => format!("{}", i),
                                _ => "N/A".to_string(),
                            };
                            summary.push(format!(
                                "KLD {}: {} ({} prompts)",
                                pair, formatted_kld, formatted_prompts
                            ));
                        }
                    }
                }
            }
        }
    }

    summary
}

fn render_report_html(
    all_models_results: &HashMap<String, HashMap<String, BenchmarkResult>>,
    post_execute_results: &HashMap<String, BenchmarkResult>,
) -> Result<String> {
    let input = build_report_input(all_models_results, post_execute_results);
    let ctx = ReportContext { input: &input };
    HtmlReportGenerator.generate(&ctx)
}

fn render_markdown_report(
    all_models_results: &HashMap<String, HashMap<String, BenchmarkResult>>,
    post_execute_results: &HashMap<String, BenchmarkResult>,
) -> Result<String> {
    let input = build_report_input(all_models_results, post_execute_results);
    let ctx = ReportContext { input: &input };
    MarkdownReportGenerator.generate(&ctx)
}

fn render_console_report(
    all_models_results: &HashMap<String, HashMap<String, BenchmarkResult>>,
    post_execute_results: &HashMap<String, BenchmarkResult>,
) -> Result<String> {
    let input = build_report_input(all_models_results, post_execute_results);
    let ctx = ReportContext { input: &input };
    ConsoleReportGenerator.generate(&ctx)
}

/// Filter the in-memory results to only include models in the comparison.
fn filter_comparison_models(
    all_models_results: &HashMap<String, HashMap<String, BenchmarkResult>>,
    comparison: &Comparison,
) -> HashMap<String, HashMap<String, BenchmarkResult>> {
    let model_names: Vec<String> = comparison.models.clone();
    if model_names.is_empty() {
        return all_models_results.clone();
    }

    all_models_results
        .iter()
        .filter(|(name, _)| model_names.contains(name))
        .map(|(name, results)| (name.clone(), results.clone()))
        .collect()
}

/// Generate all reports (HTML, Markdown, comparison reports) from in-memory results.
pub fn generate_reports(
    all_models_results: &HashMap<String, HashMap<String, BenchmarkResult>>,
    output_dir: &Path,
    comparisons: &[Comparison],
    post_execute_results: &HashMap<String, BenchmarkResult>,
) -> Result<()> {
    // Main HTML report
    let html = render_report_html(all_models_results, post_execute_results)?;
    fs::write(output_dir.join("benchmark_report.html"), html)?;
    println!("HTML report: benchmark_report.html");

    // Markdown report
    let md = render_markdown_report(all_models_results, post_execute_results)?;
    fs::write(output_dir.join("benchmark_report.md"), md)?;
    println!("Markdown report: benchmark_report.md");

    // Save raw results as JSON (for backwards compatibility)
    let raw_json = build_raw_results_json(all_models_results);
    let json = serde_json::to_string_pretty(&raw_json)?;
    fs::write(output_dir.join("results.json"), json)?;
    println!("Raw results: results.json");

    // Per-comparison reports
    for (idx, comparison) in comparisons.iter().enumerate() {
        if comparison.models.is_empty() {
            continue;
        }
        let slug = crate::utils::slugify(&comparison.title);
        let filename = if slug.is_empty() {
            format!("comparison-{}.html", idx)
        } else {
            format!("comparison-{}.html", slug)
        };
        generate_comparison_report(
            all_models_results,
            output_dir,
            &filename,
            comparison,
            post_execute_results,
        )?;
    }
    Ok(())
}

/// Generate a comparison report HTML file containing only the specified models,
/// filtered from the in-memory results.
pub fn generate_comparison_report(
    all_models_results: &HashMap<String, HashMap<String, BenchmarkResult>>,
    output_dir: &Path,
    filename: &str,
    comparison: &Comparison,
    post_execute_results: &HashMap<String, BenchmarkResult>,
) -> Result<()> {
    let filtered_models = filter_comparison_models(all_models_results, comparison);
    let filtered_post = filter_post_execute_results(post_execute_results, comparison);
    let html = render_report_html(&filtered_models, &filtered_post)?;
    let filepath = output_dir.join(filename);
    fs::write(&filepath, html)?;
    println!("Comparison report: {}", filepath.display());
    Ok(())
}

/// Filter post-execute results to include only models relevant to the comparison.
fn filter_post_execute_results(
    post_execute_results: &HashMap<String, BenchmarkResult>,
    comparison: &Comparison,
) -> HashMap<String, BenchmarkResult> {
    let model_names: Vec<String> = comparison.models.clone();
    if model_names.is_empty() {
        return post_execute_results.clone();
    }

    post_execute_results
        .iter()
        .map(|(bench_name, result)| {
            // For benchmarks like KLD that have a breakdown table with pairwise scores,
            // we need to filter the breakdown rows to only include pairs from the comparison.
            let filtered_result = if bench_name == "kld" {
                let mut filtered = result.clone();
                if let Some(breakdown) = filtered.breakdowns.get_mut("pairwise_kld") {
                    // Filter pairwise rows to only include pairs where both models are in the comparison
                    let filtered_rows = breakdown
                        .rows
                        .iter()
                        .filter(|(key, _)| {
                            if let Some((a, b)) = key.split_once('_') {
                                model_names.contains(&a.to_string())
                                    && model_names.contains(&b.to_string())
                            } else {
                                false
                            }
                        })
                        .map(|(k, v)| (k.clone(), v.clone()))
                        .collect();
                    breakdown.rows = filtered_rows;
                    // Also filter avg_kld_to_others from the raw field
                    if let Some(raw_obj) = result.raw.as_object() {
                        let filtered_avg = raw_obj
                            .get("avg_kld_to_others")
                            .and_then(|v| v.as_object())
                            .map(|avg| {
                                avg.iter()
                                    .filter(|(name, _)| model_names.contains(name))
                                    .map(|(k, v)| (k.clone(), v.clone()))
                                    .collect::<serde_json::Map<_, _>>()
                            });
                        if let Some(filtered_avg) = filtered_avg {
                            filtered.raw["avg_kld_to_others"] =
                                serde_json::Value::Object(filtered_avg);
                        }
                    }
                }
                filtered
            } else {
                result.clone()
            };
            (bench_name.clone(), filtered_result)
        })
        .collect()
}
