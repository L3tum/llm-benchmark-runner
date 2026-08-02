use super::model::{BenchmarkResult, BreakdownTable, Score, ScoreUnit};
use anyhow::Result;
use std::collections::BTreeMap;

/// Build a benchmark result with overall accuracy/pass@1 and per-category breakdown.
///
/// Extracts per-task arrays from the raw benchmark result, computes overall scores,
/// and builds a per-category breakdown table (e.g., by difficulty or by language).
///
/// # Arguments
/// * `b` - The raw benchmark result from the runner
/// * `primary_score_name` - Name for the primary metric (e.g., "accuracy" or "pass@1")
/// * `breakdown_category` - Category key for the breakdown table (e.g., "By Difficulty")
/// * `breakdown_title` - Human-readable title for the breakdown table
/// * `count_key_name` - Name for the count metric (e.g., "correct" or "passed").
///   Use `None` to default to "correct".
///
/// # Example usage
/// For a translation benchmark with difficulty levels:
/// ```notest
/// build_accuracy_report(&raw, "accuracy", "By Difficulty", "Accuracy by Difficulty Level", None)?;
/// ```
///
/// For a coding benchmark with languages:
/// ```notest
/// build_accuracy_report(&raw, "pass@1", "By Language", "Pass@1 by Programming Language", Some("passed"))?;
/// ```
pub fn build_accuracy_report(
    b: &BenchmarkResult,
    primary_score_name: &str,
    breakdown_category: &str,
    breakdown_title: &str,
    count_key_name: Option<&str>,
) -> Result<BenchmarkResult> {
    let raw = &b.raw;

    // Extract per-task results and compute totals (single-pass)
    let (total, correct, output_tokens, thinking_tokens, per_task) = {
        if let Some(pt) = raw.get("per_task").and_then(|v| v.as_array()) {
            let total = pt.len() as i64;
            let correct = pt
                .iter()
                .filter(|t| t.get("passed").and_then(|v| v.as_bool()).unwrap_or(false))
                .count() as i64;
            let out: u64 = pt
                .iter()
                .filter_map(|t| t.get("output_tokens").and_then(|v| v.as_u64()))
                .sum();
            let think: u64 = pt
                .iter()
                .filter_map(|t| t.get("thinking_tokens").and_then(|v| v.as_u64()))
                .sum();
            (total, correct, out, think, Some(pt))
        } else {
            (
                raw.get("total_tasks").and_then(|v| v.as_i64()).unwrap_or(0),
                raw.get("passed_tasks")
                    .and_then(|v| v.as_i64())
                    .unwrap_or(0),
                raw.get("output_tokens")
                    .and_then(|v| v.as_u64())
                    .unwrap_or(0),
                raw.get("thinking_tokens")
                    .and_then(|v| v.as_u64())
                    .unwrap_or(0),
                None,
            )
        }
    };

    let overall_rate = if total > 0 {
        correct as f64 / total as f64 * 100.0
    } else {
        0.0
    };

    let mut scores = BTreeMap::new();
    scores.insert(
        primary_score_name.to_string(),
        Score::float(overall_rate, ScoreUnit::Percent)
            .primary(true)
            .higher_is_better(true),
    );
    scores.insert("total".to_string(), Score::integer(total, ScoreUnit::Count));
    scores.insert(
        count_key_name.unwrap_or("correct").to_string(),
        Score::integer(correct, ScoreUnit::Count).higher_is_better(true),
    );
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

    // Per-category breakdown
    let mut breakdowns = BTreeMap::new();
    if let Some(per_task) = per_task {
        let mut cat_counts: BTreeMap<String, (i64, i64)> = BTreeMap::new();

        for task in per_task {
            // Category is stored as the first element of the categories array
            let category = task
                .get("categories")
                .and_then(|v| v.as_array())
                .and_then(|arr| arr.first())
                .and_then(|v| v.as_str())
                .unwrap_or("unknown")
                .to_string();
            let passed = task
                .get("passed")
                .and_then(|v| v.as_bool())
                .unwrap_or(false);

            let (p, t) = cat_counts.entry(category).or_insert((0, 0));
            *t += 1;
            if passed {
                *p += 1;
            }
        }

        if !cat_counts.is_empty() {
            let mut rows = BTreeMap::new();
            for (category, (passed, total_cat)) in &cat_counts {
                let rate = if *total_cat > 0 {
                    *passed as f64 / *total_cat as f64 * 100.0
                } else {
                    0.0
                };
                rows.insert(
                    category.clone(),
                    BTreeMap::from([
                        (
                            primary_score_name.to_string(),
                            Score::float(rate, ScoreUnit::Percent).display(format!("{:.1}%", rate)),
                        ),
                        (
                            "instances".to_string(),
                            Score::integer(*total_cat, ScoreUnit::Count)
                                .display(format!("{}/{}", passed, total_cat)),
                        ),
                    ]),
                );
            }
            breakdowns.insert(
                breakdown_category.to_string(),
                BreakdownTable {
                    title: breakdown_title.to_string(),
                    rows,
                },
            );
        }
    }

    Ok(BenchmarkResult {
        scores,
        breakdowns,
        error_classification: BTreeMap::new(),
        artifacts: vec![],
        diagnostics: vec![crate::reports::model::Diagnostic {
            level: "info".to_string(),
            message: format!(
                "{} = {:.1}% ({} / {})",
                primary_score_name, overall_rate, correct, total
            ),
        }],
        raw: raw.clone(),
    })
}

#[cfg(test)]
mod tests {
    use super::super::model::ScoreValue;
    use super::*;

    fn make_raw_result(tasks: Vec<(&str, bool, Option<&str>)>) -> serde_json::Value {
        serde_json::json!({
            "per_task": tasks.iter().map(|(id, passed, cat)| {
                serde_json::json!({
                    "task_id": id,
                    "passed": passed,
                    "categories": cat.map(|c| vec![c]).unwrap_or_default(),
                })
            }).collect::<Vec<_>>(),
        })
    }

    #[test]
    fn build_accuracy_report_with_per_task() {
        let raw = make_raw_result(vec![
            ("t1", true, Some("easy")),
            ("t2", false, Some("easy")),
            ("t3", true, Some("hard")),
            ("t4", true, Some("hard")),
        ]);
        let b = BenchmarkResult {
            scores: BTreeMap::new(),
            breakdowns: BTreeMap::new(),
            error_classification: BTreeMap::new(),
            artifacts: vec![],
            diagnostics: vec![],
            raw: raw.clone(),
        };

        let result =
            build_accuracy_report(&b, "accuracy", "By Category", "By Category", None).unwrap();

        assert_eq!(
            result.scores.get("accuracy").unwrap().value,
            ScoreValue::Float(75.0)
        );
        assert_eq!(
            result.scores.get("total").unwrap().value,
            ScoreValue::Integer(4)
        );
        assert_eq!(
            result.scores.get("correct").unwrap().value,
            ScoreValue::Integer(3)
        );

        let breakdown = result.breakdowns.get("By Category").unwrap();
        assert_eq!(breakdown.title, "By Category");
        assert!(breakdown.rows.contains_key("easy"));
        assert!(breakdown.rows.contains_key("hard"));
    }

    #[test]
    fn build_accuracy_report_fallback_no_per_task() {
        let raw = serde_json::json!({
            "total_tasks": 10,
            "passed_tasks": 7,
        });
        let b = BenchmarkResult {
            scores: BTreeMap::new(),
            breakdowns: BTreeMap::new(),
            error_classification: BTreeMap::new(),
            artifacts: vec![],
            diagnostics: vec![],
            raw: raw.clone(),
        };

        let result =
            build_accuracy_report(&b, "pass@1", "By Language", "Pass@1 by Language", None).unwrap();

        assert_eq!(
            result.scores.get("pass@1").unwrap().value,
            ScoreValue::Float(70.0)
        );
        assert_eq!(
            result.scores.get("total").unwrap().value,
            ScoreValue::Integer(10)
        );
        assert_eq!(
            result.scores.get("correct").unwrap().value,
            ScoreValue::Integer(7)
        );
        assert!(result.breakdowns.is_empty());
    }

    #[test]
    fn build_accuracy_report_empty() {
        let raw = serde_json::json!({ "per_task": [] });
        let b = BenchmarkResult {
            scores: BTreeMap::new(),
            breakdowns: BTreeMap::new(),
            error_classification: BTreeMap::new(),
            artifacts: vec![],
            diagnostics: vec![],
            raw: raw.clone(),
        };

        let result =
            build_accuracy_report(&b, "accuracy", "By Category", "By Category", None).unwrap();

        assert_eq!(
            result.scores.get("accuracy").unwrap().value,
            ScoreValue::Float(0.0)
        );
    }

    #[test]
    fn build_accuracy_report_missing_passed_key() {
        // per_task entries without "passed" key should count as failed
        let raw = serde_json::json!({
            "per_task": [
                {"output_tokens": 100},
                {"passed": true, "output_tokens": 50}
            ]
        });
        let b = BenchmarkResult {
            scores: BTreeMap::new(),
            breakdowns: BTreeMap::new(),
            error_classification: BTreeMap::new(),
            artifacts: vec![],
            diagnostics: vec![],
            raw: raw.clone(),
        };
        let result = build_accuracy_report(&b, "score", "By Category", "Test", None).unwrap();
        assert_eq!(
            result.scores.get("total").unwrap().value,
            ScoreValue::Integer(2)
        );
        assert_eq!(
            result.scores.get("correct").unwrap().value,
            ScoreValue::Integer(1)
        );
    }

    #[test]
    fn build_accuracy_report_non_bool_passed() {
        let raw = serde_json::json!({
            "per_task": [
                {"passed": "true"},  // string, not bool
                {"passed": 1}        // number, not bool
            ]
        });
        // Should treat non-bool as false
        let b = BenchmarkResult {
            scores: BTreeMap::new(),
            breakdowns: BTreeMap::new(),
            error_classification: BTreeMap::new(),
            artifacts: vec![],
            diagnostics: vec![],
            raw: raw.clone(),
        };
        let result = build_accuracy_report(&b, "score", "By Category", "Test", None).unwrap();
        assert_eq!(
            result.scores.get("correct").unwrap().value,
            ScoreValue::Integer(0)
        );
    }
}
