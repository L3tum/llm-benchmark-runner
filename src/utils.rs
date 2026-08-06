use std::collections::BTreeMap;
use std::time::Duration;

use crate::shared::{BenchmarkResult, Diagnostic, Score, ScoreUnit};

/// Extract common task stats from a raw benchmark result JSON value.
/// Returns `(total, correct, output_tokens, thinking_tokens)`.
pub fn extract_task_stats(raw: &serde_json::Value) -> (i64, i64, u64, u64) {
    let total = raw.get("total_tasks").and_then(|v| v.as_i64()).unwrap_or(0);
    let correct = raw
        .get("passed_tasks")
        .and_then(|v| v.as_i64())
        .unwrap_or(0);
    let output_tokens = raw
        .get("output_tokens")
        .and_then(|v| v.as_u64())
        .unwrap_or(0);
    let thinking_tokens = raw
        .get("thinking_tokens")
        .and_then(|v| v.as_u64())
        .unwrap_or(0);
    (total, correct, output_tokens, thinking_tokens)
}

/// Build a standard accuracy-style `BenchmarkResult` from a raw payload, computing
/// the `accuracy` (percent) score and total/correct/token scores. Shared by the
/// string-manipulation benchmarks (base64/hex/morse/reverse) to avoid copy-paste.
pub fn build_accuracy_result(label: &str, b: &BenchmarkResult) -> anyhow::Result<BenchmarkResult> {
    let (total, correct, output_tokens, thinking_tokens) = extract_task_stats(&b.raw);
    let accuracy = if total > 0 {
        correct as f64 / total as f64 * 100.0
    } else {
        0.0
    };

    let mut scores = BTreeMap::new();
    scores.insert(
        "accuracy".to_string(),
        Score::float(accuracy, ScoreUnit::Percent)
            .primary(true)
            .higher_is_better(true),
    );
    scores.insert("total".to_string(), Score::integer(total, ScoreUnit::Count));
    scores.insert(
        "correct".to_string(),
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

    Ok(BenchmarkResult {
        scores,
        breakdowns: BTreeMap::new(),
        error_classification: BTreeMap::new(),
        artifacts: vec![],
        diagnostics: vec![Diagnostic {
            level: "info".to_string(),
            message: format!("{}: {} correct out of {}", label, correct, total),
        }],
        raw: b.raw.clone(),
    })
}

/// Format a Duration as "H:MM:SS" or "MM:SS"
pub fn format_duration(d: Duration) -> String {
    let secs = d.as_secs();
    if secs >= 3600 {
        let h = secs / 3600;
        let m = (secs % 3600) / 60;
        let s = secs % 60;
        format!("{}:{:02}:{:02}", h, m, s)
    } else {
        let m = secs / 60;
        let s = secs % 60;
        format!("{:01}:{:02}", m, s)
    }
}

/// Convert a human-readable title into a URL-safe slug (e.g., "Q4 vs Q5" → "q4-vs-q5")
pub fn slugify(title: &str) -> String {
    title
        .to_lowercase()
        .chars()
        .filter(|c| c.is_alphanumeric() || c.is_whitespace() || *c == '-')
        .map(|c| if c.is_whitespace() { '-' } else { c })
        .collect::<String>()
        .split('-')
        .filter(|s| !s.is_empty())
        .collect::<Vec<_>>()
        .join("-")
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::shared::{BenchmarkResult, ScoreValue};

    fn raw_result(passed: i64, total: i64, output: u64, think: u64) -> serde_json::Value {
        serde_json::json!({
            "passed_tasks": passed,
            "total_tasks": total,
            "output_tokens": output,
            "thinking_tokens": think,
        })
    }

    #[test]
    fn build_accuracy_result_computes_percent_and_labels() {
        let b = BenchmarkResult {
            scores: Default::default(),
            breakdowns: Default::default(),
            error_classification: Default::default(),
            artifacts: vec![],
            diagnostics: vec![],
            raw: raw_result(3, 4, 100, 50),
        };
        let out = build_accuracy_result("Base64", &b).unwrap();
        let acc = &out.scores["accuracy"];
        assert_eq!(acc.value, ScoreValue::Float(75.0)); // 3/4 * 100
        assert_eq!(out.scores["total"].value, ScoreValue::Integer(4));
        assert_eq!(out.scores["correct"].value, ScoreValue::Integer(3));
        assert_eq!(out.scores["output_tokens"].value, ScoreValue::Integer(100));
        assert_eq!(out.scores["thinking_tokens"].value, ScoreValue::Integer(50));
        assert_eq!(out.diagnostics[0].message, "Base64: 3 correct out of 4");
    }

    #[test]
    fn build_accuracy_result_zero_total_is_zero() {
        let b = BenchmarkResult {
            scores: Default::default(),
            breakdowns: Default::default(),
            error_classification: Default::default(),
            artifacts: vec![],
            diagnostics: vec![],
            raw: raw_result(0, 0, 0, 0),
        };
        let out = build_accuracy_result("Hex", &b).unwrap();
        assert_eq!(out.scores["accuracy"].value, ScoreValue::Float(0.0));
    }
}
