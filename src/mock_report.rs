//! Mock report generation for testing HTML/Markdown report rendering.
//!
//! Creates mock benchmark results and generates sample reports.

use anyhow::Result;
use std::collections::{BTreeMap, HashMap};

use llm_benchmark_runner::reports::generator::ReportGenerator;
use llm_benchmark_runner::reports::model::{Artifact, BenchmarkResult, Score, ScoreUnit};

/// Generates a mock report with two models having multi-building minebench outputs.
pub fn generate() -> Result<()> {
    // Print to confirm this function runs
    eprintln!("Generating mock report...");

    let output_dir = "benchmark_results";
    std::fs::create_dir_all(output_dir)?;

    // Write mock output files with multi-building structure
    let mock_model_a_output = serde_json::json!({
        "models": {
            "MockModel-A": {
                "buildings": {
                    "tree": {
                        "version": "1.0",
                        "blocks": [
                            {"x": 0, "y": 0, "z": 0, "type": "stone"},
                            {"x": 1, "y": 0, "z": 0, "type": "stone"},
                            {"x": 2, "y": 0, "z": 0, "type": "stone"},
                            {"x": 0, "y": 1, "z": 0, "type": "oak_planks"},
                            {"x": 1, "y": 1, "z": 0, "type": "oak_planks"},
                            {"x": 2, "y": 1, "z": 0, "type": "oak_planks"},
                            {"x": 0, "y": 2, "z": 0, "type": "oak_planks"},
                            {"x": 1, "y": 2, "z": 0, "type": "oak_leaves"},
                            {"x": 2, "y": 2, "z": 0, "type": "oak_planks"},
                            {"x": 1, "y": 3, "z": 0, "type": "oak_leaves"},
                            {"x": 1, "y": 4, "z": 0, "type": "oak_leaves"},
                            {"x": 0, "y": 5, "z": 0, "type": "stone"},
                            {"x": 1, "y": 5, "z": 0, "type": "stone"},
                            {"x": 2, "y": 5, "z": 0, "type": "stone"}
                        ]
                    },
                    "tower": {
                        "version": "1.0",
                        "blocks": [
                            {"x": 0, "y": 0, "z": 0, "type": "cobblestone"},
                            {"x": 1, "y": 0, "z": 0, "type": "cobblestone"},
                            {"x": 2, "y": 0, "z": 0, "type": "cobblestone"},
                            {"x": 1, "y": 1, "z": 0, "type": "cobblestone"},
                            {"x": 1, "y": 2, "z": 0, "type": "cobblestone"},
                            {"x": 1, "y": 3, "z": 0, "type": "stone"},
                            {"x": 0, "y": 4, "z": 0, "type": "stone"},
                            {"x": 1, "y": 4, "z": 0, "type": "stone"},
                            {"x": 2, "y": 4, "z": 0, "type": "stone"}
                        ]
                    }
                }
            }
        }
    });
    let mock_model_b_output = serde_json::json!({
        "models": {
            "MockModel-B": {
                "buildings": {
                    "house": {
                        "version": "1.0",
                        "blocks": [
                            {"x": 0, "y": 0, "z": 0, "type": "oak_planks"},
                            {"x": 1, "y": 0, "z": 0, "type": "oak_planks"},
                            {"x": 2, "y": 0, "z": 0, "type": "oak_planks"},
                            {"x": 3, "y": 0, "z": 0, "type": "oak_planks"},
                            {"x": 0, "y": 1, "z": 0, "type": "stone_bricks"},
                            {"x": 3, "y": 1, "z": 0, "type": "stone_bricks"},
                            {"x": 0, "y": 2, "z": 0, "type": "stone_bricks"},
                            {"x": 3, "y": 2, "z": 0, "type": "stone_bricks"},
                            {"x": 0, "y": 3, "z": 0, "type": "stone_bricks"},
                            {"x": 3, "y": 3, "z": 0, "type": "stone_bricks"},
                            {"x": 1, "y": 3, "z": 0, "type": "stone_bricks"},
                            {"x": 2, "y": 3, "z": 0, "type": "stone_bricks"},
                            {"x": 1, "y": 2, "z": 0, "type": "glass"},
                            {"x": 2, "y": 2, "z": 0, "type": "glass"}
                        ]
                    }
                }
            }
        }
    });

    let path_a = format!("{}/mock_model_a_output.json", output_dir);
    let path_b = format!("{}/mock_model_b_output.json", output_dir);
    std::fs::write(&path_a, serde_json::to_string_pretty(&mock_model_a_output)?)?;
    std::fs::write(&path_b, serde_json::to_string_pretty(&mock_model_b_output)?)?;

    // Create mock minebench results for Model A (2 buildings: tree, tower)
    let mut minebench_scores_a = BTreeMap::new();
    minebench_scores_a.insert(
        "valid_buildings".to_string(),
        Score::integer(2, ScoreUnit::Count),
    );
    minebench_scores_a.insert(
        "total_buildings".to_string(),
        Score::integer(2, ScoreUnit::Count),
    );
    minebench_scores_a.insert(
        "output_tokens".to_string(),
        Score::integer(5000, ScoreUnit::Tokens),
    );
    minebench_scores_a.insert(
        "thinking_tokens".to_string(),
        Score::integer(2000, ScoreUnit::Tokens),
    );

    let minebench_result_a = BenchmarkResult {
        scores: minebench_scores_a,
        breakdowns: BTreeMap::new(),
        error_classification: BTreeMap::new(),
        artifacts: vec![Artifact {
            label: "Output".to_string(),
            path: path_a.clone(),
            kind: "file".to_string(),
        }],
        diagnostics: vec![],
        raw: serde_json::json!({
            "valid_buildings": 2,
            "total_buildings": 2,
            "output_file": path_a,
            "output_tokens": 5000,
            "thinking_tokens": 2000
        }),
    };

    // Create mock minebench results for Model B (1 building: house)
    let mut minebench_scores_b = BTreeMap::new();
    minebench_scores_b.insert(
        "valid_buildings".to_string(),
        Score::integer(1, ScoreUnit::Count),
    );
    minebench_scores_b.insert(
        "total_buildings".to_string(),
        Score::integer(1, ScoreUnit::Count),
    );
    minebench_scores_b.insert(
        "output_tokens".to_string(),
        Score::integer(8000, ScoreUnit::Tokens),
    );
    minebench_scores_b.insert(
        "thinking_tokens".to_string(),
        Score::integer(3000, ScoreUnit::Tokens),
    );

    let minebench_result_b = BenchmarkResult {
        scores: minebench_scores_b,
        breakdowns: BTreeMap::new(),
        error_classification: BTreeMap::new(),
        artifacts: vec![Artifact {
            label: "Output".to_string(),
            path: path_b.clone(),
            kind: "file".to_string(),
        }],
        diagnostics: vec![],
        raw: serde_json::json!({
            "valid_buildings": 1,
            "total_buildings": 1,
            "output_file": path_b,
            "output_tokens": 8000,
            "thinking_tokens": 3000
        }),
    };

    // Create a mock kld result
    let mut kld_scores_a = BTreeMap::new();
    kld_scores_a.insert(
        "avg_kld_to_others".to_string(),
        Score::float(1.23, ScoreUnit::Kld),
    );
    let kld_result_a = BenchmarkResult {
        scores: kld_scores_a,
        breakdowns: BTreeMap::new(),
        error_classification: BTreeMap::new(),
        artifacts: vec![],
        diagnostics: vec![],
        raw: serde_json::Value::Null,
    };

    let mut kld_scores_b = BTreeMap::new();
    kld_scores_b.insert(
        "avg_kld_to_others".to_string(),
        Score::float(1.45, ScoreUnit::Kld),
    );
    let kld_result_b = BenchmarkResult {
        scores: kld_scores_b,
        breakdowns: BTreeMap::new(),
        error_classification: BTreeMap::new(),
        artifacts: vec![],
        diagnostics: vec![],
        raw: serde_json::Value::Null,
    };

    // Create report input
    let mut all_models_results = HashMap::new();
    let mut models_a = HashMap::new();
    models_a.insert("minebench".to_string(), minebench_result_a);
    models_a.insert("kld".to_string(), kld_result_a);
    all_models_results.insert("MockModel-A".to_string(), models_a);

    let mut models_b = HashMap::new();
    models_b.insert("minebench".to_string(), minebench_result_b);
    models_b.insert("kld".to_string(), kld_result_b);
    all_models_results.insert("MockModel-B".to_string(), models_b);

    let post_execute_results = HashMap::new();

    let report_input = llm_benchmark_runner::report::build_report_input(
        &all_models_results,
        &post_execute_results,
    );
    let output_dir = "benchmark_results";

    let html_generator = llm_benchmark_runner::reports::html::HtmlReportGenerator;
    let ctx = llm_benchmark_runner::reports::generator::ReportContext {
        input: &report_input,
    };
    let html = html_generator.generate(&ctx)?;
    let output_path = format!("{}/mock_report.html", output_dir);
    std::fs::write(&output_path, html)?;
    println!("Mock HTML report generated: {}", output_path);

    let md_generator = llm_benchmark_runner::reports::markdown::MarkdownReportGenerator;
    let md = md_generator.generate(&ctx)?;
    let md_path = format!("{}/mock_report.md", output_dir);
    std::fs::write(&md_path, md)?;

    // Also save the raw results
    let raw_results = serde_json::to_string_pretty(&report_input.raw_results)?;
    let raw_path = format!("{}/mock_raw_results.json", output_dir);
    std::fs::write(&raw_path, raw_results)?;

    Ok(())
}
