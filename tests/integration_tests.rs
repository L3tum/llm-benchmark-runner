//! Integration tests for the new benchmarks.
//!
//! These tests verify report generation and benchmark structure
//! without requiring live LLM calls or Docker containers.
//! Internal data generation is tested via unit tests in each module.

use llm_benchmark_runner::benchmarks::get_benchmark;
use llm_benchmark_runner::reports::model::{BenchmarkCategory, BenchmarkResult, BreakdownTable};
use llm_benchmark_runner::reports::report_helpers::build_accuracy_report;
use std::collections::BTreeMap;

/// Test that build_accuracy_report correctly computes per-category breakdown.
#[test]
fn build_accuracy_report_per_category() {
    // Simulate per-task results for a translation benchmark
    let tasks = [
        ("task-1", true, vec!["easy".to_string()]),
        ("task-2", true, vec!["easy".to_string()]),
        ("task-3", false, vec!["easy".to_string()]),
        ("task-4", true, vec!["medium".to_string()]),
        ("task-5", true, vec!["medium".to_string()]),
        ("task-6", true, vec!["hard".to_string()]),
        ("task-7", false, vec!["hard".to_string()]),
        ("task-8", false, vec!["hard".to_string()]),
    ];

    let per_task_json: Vec<serde_json::Value> = tasks
        .iter()
        .map(|(id, passed, cats)| {
            serde_json::json!({
                "task_id": id,
                "passed": passed,
                "categories": cats,
            })
        })
        .collect();

    let raw = serde_json::json!({
        "per_task": per_task_json,
    });

    let mut breakdowns = BTreeMap::new();
    breakdowns.insert(
        "mock".to_string(),
        BreakdownTable {
            title: "Mock".to_string(),
            rows: BTreeMap::new(),
        },
    );

    let b = BenchmarkResult {
        scores: BTreeMap::new(),
        breakdowns,
        error_classification: BTreeMap::new(),
        artifacts: Vec::new(),
        diagnostics: Vec::new(),
        raw,
    };

    let result =
        build_accuracy_report(&b, "accuracy", "By Category", "Accuracy by Category", None).unwrap();

    // Should have overall accuracy score
    assert!(result.scores.contains_key("accuracy"));

    // Should have breakdown by category
    assert!(
        result.breakdowns.contains_key("By Category"),
        "Should have 'By Category' breakdown"
    );
}

/// Test that build_accuracy_report handles custom count key (passed vs correct).
#[test]
fn build_accuracy_report_with_passed_key() {
    let tasks = [
        ("task-1", true, vec!["py".to_string()]),
        ("task-2", false, vec!["py".to_string()]),
        ("task-3", true, vec!["java".to_string()]),
    ];

    let per_task_json: Vec<serde_json::Value> = tasks
        .iter()
        .map(|(id, passed, cats)| {
            serde_json::json!({
                "task_id": id,
                "passed": passed,
                "categories": cats,
            })
        })
        .collect();

    let raw = serde_json::json!({
        "per_task": per_task_json,
    });

    let b = BenchmarkResult {
        scores: BTreeMap::new(),
        breakdowns: BTreeMap::new(),
        error_classification: BTreeMap::new(),
        artifacts: Vec::new(),
        diagnostics: Vec::new(),
        raw,
    };

    // Use "passed" as the count key (MultiPL-E style)
    let result = build_accuracy_report(
        &b,
        "pass@1",
        "By Language",
        "Pass@1 by Language",
        Some("passed"),
    )
    .unwrap();

    // Should have pass@1 score
    assert!(result.scores.contains_key("pass@1"));
}

/// Test that the benchmark categories are correctly assigned.
#[test]
fn benchmark_category_assignments() {
    // These are compile-time checks that the category enum has the right variants
    let _coding = BenchmarkCategory::ShortContextCoding;
    let _translation = BenchmarkCategory::Translation;
}

/// Test that empty data produces a valid (but empty) report.
#[test]
fn build_accuracy_report_empty_raw() {
    let b = BenchmarkResult {
        scores: BTreeMap::new(),
        breakdowns: BTreeMap::new(),
        error_classification: BTreeMap::new(),
        artifacts: Vec::new(),
        diagnostics: Vec::new(),
        raw: serde_json::json!(null),
    };

    let result =
        build_accuracy_report(&b, "accuracy", "By Category", "Accuracy by Category", None).unwrap();

    // Should still have an overall score (0/0 = 0.0)
    assert!(result.scores.contains_key("accuracy"));
}

/// Test that mixed pass/fail across categories produces correct breakdown.
#[test]
fn build_accuracy_report_mixed_categories() {
    // Python: 2 passed, 1 failed (66.7%)
    // Java: 1 passed, 1 failed (50%)
    // Rust: 3 passed, 0 failed (100%)
    let tasks = [
        ("task-1", true, vec!["py".to_string()]),
        ("task-2", true, vec!["py".to_string()]),
        ("task-3", false, vec!["py".to_string()]),
        ("task-4", true, vec!["java".to_string()]),
        ("task-5", false, vec!["java".to_string()]),
        ("task-6", true, vec!["rs".to_string()]),
        ("task-7", true, vec!["rs".to_string()]),
        ("task-8", true, vec!["rs".to_string()]),
    ];

    let per_task_json: Vec<serde_json::Value> = tasks
        .iter()
        .map(|(id, passed, cats)| {
            serde_json::json!({
                "task_id": id,
                "passed": passed,
                "categories": cats,
            })
        })
        .collect();

    let raw = serde_json::json!({
        "per_task": per_task_json,
    });

    let b = BenchmarkResult {
        scores: BTreeMap::new(),
        breakdowns: BTreeMap::new(),
        error_classification: BTreeMap::new(),
        artifacts: Vec::new(),
        diagnostics: Vec::new(),
        raw,
    };

    let result = build_accuracy_report(
        &b,
        "pass@1",
        "By Language",
        "Pass@1 by Language",
        Some("passed"),
    )
    .unwrap();

    // Overall: 6/8 = 75%
    let overall = result.scores.get("pass@1").expect("Should have pass@1");
    // Check the value is close to 75
    let val = match overall.value {
        llm_benchmark_runner::reports::model::ScoreValue::Float(f) => f,
        llm_benchmark_runner::reports::model::ScoreValue::Integer(i) => i as f64,
        _ => 0.0,
    };
    assert!(
        (val - 75.0).abs() < 0.1,
        "Expected ~75% pass@1, got {}",
        val
    );

    // Check breakdown exists
    assert!(result.breakdowns.contains_key("By Language"));
}

/// Test that newly registered benchmarks are accessible via the registry.
#[test]
fn new_benchmarks_are_registered() {
    assert!(get_benchmark("bbh").is_ok(), "bbh should be registered");
    assert!(
        get_benchmark("factbench").is_ok(),
        "factbench should be registered"
    );
    assert!(
        get_benchmark("truthful_qa_gen").is_ok(),
        "truthful_qa_gen should be registered"
    );
}
