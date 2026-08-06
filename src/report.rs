use crate::config::Comparison;
use crate::reports::generator::{ReportContext, ReportGenerator};
use crate::reports::html::HtmlReportGenerator;
use crate::reports::markdown::MarkdownReportGenerator;
use crate::reports::model::{ReportInput, TestReportData};
use crate::shared::{BenchmarkResult, ScoreValue, TestName};
use anyhow::Result;
use std::collections::{BTreeMap, HashMap, HashSet};
use std::fs;
use std::path::Path;

/// Ordered list of category names for display (includes reserved empty categories).
/// Build a `ReportInput` from in-memory `BenchmarkResult` objects.
/// Only used by the mock report command. Benchmarks are discovered
/// dynamically from the registry — no manual list maintenance needed.
pub fn build_report_input(
    all_models_results: &HashMap<String, HashMap<String, BenchmarkResult>>,
    post_execute_results: &HashMap<String, BenchmarkResult>,
) -> ReportInput {
    let models_evaluated: Vec<String> = all_models_results.keys().cloned().collect();
    let timestamp = chrono::Utc::now()
        .format("%Y-%m-%d %H:%M:%S UTC")
        .to_string();

    // Build benchmark list from the registry — auto-discovers all registered benchmarks
    let benchmark_entries: Vec<_> = crate::benchmarks::iter_benchmarks().collect();

    let mut tests = BTreeMap::new();

    for (bench_name, benchmark) in &benchmark_entries {
        // Collect per-model BenchmarkResult for this benchmark
        let mut model_results: BTreeMap<String, BenchmarkResult> = BTreeMap::new();

        for (model_name, bench_results) in all_models_results {
            if let Some(bench_result) = bench_results.get(*bench_name) {
                // Call to_report_result on the in-memory result
                match benchmark.to_report_result(bench_result) {
                    Ok(result) => {
                        model_results.insert(model_name.clone(), result);
                    }
                    Err(e) => {
                        eprintln!(
                            "Warning: Failed to convert {} result for {}: {}",
                            bench_name, model_name, e
                        );
                    }
                }
            }
        }

        // Aggregate results (from post_execute)
        let aggregate = post_execute_results
            .get(*bench_name)
            .and_then(|post_result| benchmark.to_report_aggregate(post_result).ok().flatten());

        if !model_results.is_empty() || aggregate.is_some() {
            tests.insert(
                TestName::new(*bench_name),
                TestReportData {
                    name: TestName::new(*bench_name),
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
                // Percent-unit scores are already stored as 0-100 (e.g. pass_rate,
                // accuracy computed as count/total*100.0). Do NOT multiply again.
                ScoreValue::Float(f) => format!("{:.1}%", f),
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

/// Filter the in-memory results to only include models in the comparison.
fn filter_comparison_models(
    all_models_results: &HashMap<String, HashMap<String, BenchmarkResult>>,
    comparison: &Comparison,
) -> HashMap<String, HashMap<String, BenchmarkResult>> {
    if comparison.models.is_empty() {
        return all_models_results.clone();
    }

    let model_names: HashSet<&str> = comparison.models.iter().map(|s| s.as_str()).collect();

    all_models_results
        .iter()
        .filter(|(name, _)| model_names.contains(name.as_str()))
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
    let model_names: HashSet<&str> = comparison.models.iter().map(|s| s.as_str()).collect();
    if model_names.is_empty() {
        return post_execute_results.clone();
    }

    post_execute_results
        .iter()
        .map(|(bench_name, result)| {
            // For benchmarks like KLD that have a breakdown table with pairwise scores,
            // filter the breakdown rows to only include pairs from the comparison.
            // The KLD-specific logic lives in benchmarks/kld.rs (keeps this filter generic).
            let filtered_result = if bench_name == "kld" {
                crate::benchmarks::kld::filter_kld_by_models(result, &model_names)
            } else {
                result.clone()
            };
            (bench_name.clone(), filtered_result)
        })
        .collect()
}
