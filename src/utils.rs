use std::time::Duration;

/// Extract common task stats from a raw benchmark result JSON value.
/// Returns `(total, correct, output_tokens, thinking_tokens)`.
pub fn extract_task_stats(raw: &serde_json::Value) -> (i64, i64, i64, i64) {
    let total = raw.get("total_tasks").and_then(|v| v.as_i64()).unwrap_or(0);
    let correct = raw
        .get("passed_tasks")
        .and_then(|v| v.as_i64())
        .unwrap_or(0);
    let output_tokens = raw
        .get("output_tokens")
        .and_then(|v| v.as_i64())
        .unwrap_or(0);
    let thinking_tokens = raw
        .get("thinking_tokens")
        .and_then(|v| v.as_i64())
        .unwrap_or(0);
    (total, correct, output_tokens, thinking_tokens)
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
