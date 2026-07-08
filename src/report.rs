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
use std::collections::BTreeMap;
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

/// Build a `ReportInput` from the legacy results JSON, delegating extraction
/// to each benchmark's `to_report_result` and `to_report_aggregate` methods.
fn build_report_input(results: &serde_json::Value) -> Result<ReportInput> {
    let models_evaluated: Vec<String> = results
        .get("models")
        .and_then(|v| v.as_object())
        .map(|obj| obj.keys().cloned().collect())
        .unwrap_or_default();

    let timestamp = chrono::Utc::now()
        .format("%Y-%m-%d %H:%M:%S UTC")
        .to_string();

    let mut tests = BTreeMap::new();

    // For each benchmark, get its results from the top-level JSON and delegate to the benchmark.
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

    for benchmark in benchmarks {
        let raw = results
            .get(benchmark.name())
            .ok_or_else(|| anyhow::anyhow!("Missing {} results in JSON", benchmark.name()))?;

        // Model-level results
        let raw_models = raw
            .get("models")
            .cloned()
            .unwrap_or(serde_json::Value::Null);
        let model_results: BTreeMap<String, BenchmarkResult> =
            if let Some(obj) = raw_models.as_object() {
                obj.iter()
                    .filter_map(|(model, raw_result)| {
                        match benchmark.to_report_result(raw_result) {
                            Ok(result) => Some((model.clone(), result)),
                            Err(e) => {
                                eprintln!(
                                    "Warning: Failed to convert {} result for {}: {}",
                                    benchmark.name(),
                                    model,
                                    e
                                );
                                None
                            }
                        }
                    })
                    .collect()
            } else {
                BTreeMap::new()
            };

        // Aggregate (e.g., KLD pairwise)
        let aggregate = benchmark.to_report_aggregate(raw).ok().flatten();

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

    // Generate summary from the built test data
    let summary = generate_summary_from_tests(&tests);

    Ok(ReportInput {
        generated_at: timestamp,
        models: models_evaluated,
        tests,
        summary,
        raw_results: results.clone(),
    })
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

fn render_report_html(results: &serde_json::Value) -> Result<String> {
    let input = build_report_input(results)?;
    let ctx = ReportContext { input: &input };
    HtmlReportGenerator.generate(&ctx)
}

fn render_markdown_report(results: &serde_json::Value) -> Result<String> {
    let input = build_report_input(results)?;
    let ctx = ReportContext { input: &input };
    MarkdownReportGenerator.generate(&ctx)
}

fn render_console_report(results: &serde_json::Value) -> Result<String> {
    let input = build_report_input(results)?;
    let ctx = ReportContext { input: &input };
    ConsoleReportGenerator.generate(&ctx)
}

pub fn generate_reports(
    results: &serde_json::Value,
    output_dir: &Path,
    comparisons: &[Comparison],
) -> Result<()> {
    // Main HTML report
    let html = render_report_html(results)?;
    fs::write(output_dir.join("benchmark_report.html"), html)?;
    println!("HTML report: benchmark_report.html");

    // Markdown report
    let md = render_markdown_report(results)?;
    fs::write(output_dir.join("benchmark_report.md"), md)?;
    println!("Markdown report: benchmark_report.md");

    // Save raw results as JSON
    let json = serde_json::to_string_pretty(results)?;
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
        generate_comparison_report(results, output_dir, &filename, comparison)?;
    }
    Ok(())
}

/// Filter the results JSON to only include models in the comparison.
pub fn filter_comparison_results(
    results: &serde_json::Value,
    comparison: &Comparison,
) -> serde_json::Value {
    let model_names: Vec<String> = comparison.models.clone();
    if model_names.is_empty() {
        return results.clone();
    }

    let mut filtered: serde_json::Value =
        serde_json::from_str(&serde_json::to_string(results).unwrap()).unwrap();

    if let Some(models_obj) = filtered.get_mut("models").and_then(|v| v.as_object_mut()) {
        // Retain only models that are in the comparison
        models_obj.retain(|key, _| model_names.contains(key));
    }

    // Filter KLD pairwise results (legacy top-level structure)
    if let Some(kld_results) = filtered.get("kld_pairwise") {
        let kld_filtered = filter_kld_results(kld_results, &model_names);
        filtered["kld_pairwise"] = kld_filtered;
    }

    filtered
}

/// Filter KLD pairwise results to only include models in the comparison.
fn filter_kld_results(
    kld_results: &serde_json::Value,
    model_names: &[String],
) -> serde_json::Value {
    let mut filtered = serde_json::Map::new();

    if let Some(pairwise) = kld_results.get("pairwise").and_then(|v| v.as_object()) {
        let new_pairwise: serde_json::Map<String, serde_json::Value> = pairwise
            .iter()
            .filter(|(key, _)| {
                if let Some((a, b)) = key.split_once('_') {
                    model_names.contains(&a.to_string()) && model_names.contains(&b.to_string())
                } else {
                    false
                }
            })
            .map(|(k, v)| (k.clone(), v.clone()))
            .collect();
        filtered.insert(
            "pairwise".to_string(),
            serde_json::Value::Object(new_pairwise),
        );
    }

    if let Some(avg) = kld_results
        .get("avg_kld_to_others")
        .and_then(|v| v.as_object())
    {
        let filtered_avg: serde_json::Map<String, serde_json::Value> = avg
            .iter()
            .filter(|(name, _)| model_names.contains(name))
            .map(|(k, v)| (k.clone(), v.clone()))
            .collect();
        filtered.insert(
            "avg_kld_to_others".to_string(),
            serde_json::Value::Object(filtered_avg),
        );
    }

    serde_json::Value::Object(filtered)
}

/// Generate a comparison report HTML file containing only the specified models,
/// filtered from the results JSON.
pub fn generate_comparison_report(
    results: &serde_json::Value,
    output_dir: &Path,
    filename: &str,
    comparison: &Comparison,
) -> Result<()> {
    let filtered_results = filter_comparison_results(results, comparison);
    let html = render_report_html(&filtered_results)?;
    let filepath = output_dir.join(filename);
    fs::write(&filepath, html)?;
    println!("Comparison report: {}", filepath.display());
    Ok(())
}
