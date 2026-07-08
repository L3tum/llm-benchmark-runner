use super::categories::get_category_order;
use super::generator::{ReportContext, ReportGenerator};
use super::model::*;
use anyhow::Result;
use std::collections::BTreeMap;

pub struct MarkdownReportGenerator;

impl ReportGenerator for MarkdownReportGenerator {
    type Output = String;

    fn generate(&self, ctx: &ReportContext<'_>) -> Result<Self::Output> {
        let input = ctx.input;
        let mut md = String::new();

        // Header
        md.push_str(&format!(
            "# Benchmark Report — {}\n\n**Models:** {}\n",
            input.generated_at,
            input.models.join(", ")
        ));

        // Token usage summary
        // (currently token_usage is not in ReportInput — we'll add it later)

        // Per-category tables
        let categories = get_category_order();
        for cat in categories {
            // Collect tests in this category
            let tests_in_cat: Vec<_> = input
                .tests
                .values()
                .filter(|td| &td.category == cat)
                .collect();
            if tests_in_cat.is_empty() {
                continue;
            }
            md.push_str(&format!("## {}\n", cat.display()));
            for test_data in tests_in_cat {
                render_test_md(&mut md, test_data);
            }
        }

        // Summary
        if !input.summary.is_empty() {
            md.push_str("\n## Summary\n\n");
            for line in &input.summary {
                md.push_str(&format!("- {}\n", line));
            }
        }

        Ok(md)
    }
}

fn render_test_md(md: &mut String, test_data: &TestReportData) {
    md.push_str(&format!("\n### {}\n\n", test_data.display_name));

    // Generic scores table
    let _models: Vec<String> = test_data.model_results.keys().cloned().collect();
    let primary_scores: Vec<String> = test_data
        .model_results
        .values()
        .filter_map(|br| {
            br.scores
                .values()
                .find(|s| s.primary)
                .map(|s| s.display_value())
        })
        .collect();

    if primary_scores.is_empty() {
        md.push_str("**Score:** N/A\n\n");
    } else {
        // Header row
        md.push_str("| Model | Score | ");
        md.push_str(&format!("{} |", test_data.display_name));
        md.push_str("\n| --- | --- | ");
        md.push_str(&format!("{} |", test_data.display_name));
        md.push('\n');

        // Per-model row
        for (model, br) in &test_data.model_results {
            let score = br
                .scores
                .values()
                .find(|s| s.primary)
                .map(|s| s.display_value())
                .unwrap_or_else(|| "–".into());
            md.push_str(&format!(
                "| {} | {} | {} |\n",
                model, score, test_data.display_name
            ));
        }
    }

    // Token usage if available
    let token_metrics = br_token_metrics(&test_data.model_results);
    if !token_metrics.is_empty() {
        md.push_str("\n**Token usage:**\n\n");
        md.push_str("| Model | Output tokens | Thinking tokens |\n");
        md.push_str("| --- | --- | --- |\n");
        for (model, output, thinking) in &token_metrics {
            md.push_str(&format!("| {} | {} | {} |\n", model, output, thinking));
        }
        md.push('\n');
    }

    // Breakdown tables
    if let Some(first_br) = test_data.model_results.values().next() {
        for table in first_br.breakdowns.values() {
            md.push_str(&format!("\n**{}**\n\n", table.title));
            if let Some(first_row_metrics) = table.rows.values().next() {
                let metric_cols: Vec<String> = first_row_metrics.keys().cloned().collect();
                // Header
                md.push_str("| | ");
                md.push_str(&metric_cols.join(" | "));
                md.push_str(" |\n");
                md.push_str("| --- | ");
                md.push_str(
                    &metric_cols
                        .iter()
                        .map(|_| "---")
                        .collect::<Vec<_>>()
                        .join(" | "),
                );
                md.push_str(" |\n");
                // Rows per model
                for model in test_data.model_results.keys() {
                    md.push_str(&format!("| {} |", model));
                    for metrics in table.rows.values() {
                        for (_metric, score) in metric_cols.iter().zip(metrics.values()) {
                            md.push_str(&format!(" {} |", score.display_value()));
                        }
                    }
                    md.push('\n');
                }
            }
            md.push('\n');
        }
    }
}

fn br_token_metrics(
    model_results: &BTreeMap<String, BenchmarkResult>,
) -> Vec<(String, String, String)> {
    model_results
        .iter()
        .map(|(model, br)| {
            let output = br
                .scores
                .get("output_tokens")
                .map(|s| s.display_value())
                .unwrap_or_else(|| "–".into());
            let thinking = br
                .scores
                .get("thinking_tokens")
                .map(|s| s.display_value())
                .unwrap_or_else(|| "–".into());
            (model.clone(), output, thinking)
        })
        .collect()
}
