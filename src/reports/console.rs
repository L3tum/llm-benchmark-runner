use super::categories::get_category_order;
use super::generator::{ConsoleReport, ReportContext, ReportGenerator};
use super::model::*;
use anyhow::Result;

pub struct ConsoleReportGenerator;

impl ReportGenerator for ConsoleReportGenerator {
    type Output = String;

    fn generate(&self, ctx: &ReportContext<'_>) -> Result<Self::Output> {
        let input = ctx.input;
        let mut out = String::new();

        out.push_str(&format!(
            "\n=== Benchmark Report ({}) ===\nModels: {}\n",
            input.generated_at,
            input.models.join(", ")
        ));

        let categories = get_category_order();
        for cat in categories {
            let tests: Vec<_> = input
                .tests
                .values()
                .filter(|td| &td.category == cat)
                .collect();
            if tests.is_empty() {
                continue;
            }
            out.push_str(&format!("\n[{}]\n", cat.display()));
            for test_data in tests {
                render_test_console(&mut out, test_data);
            }
        }

        // Summary
        if !input.summary.is_empty() {
            out.push_str("\n--- Summary ---\n");
            for line in &input.summary {
                out.push_str(&format!("{}\n", line));
            }
        }

        Ok(out)
    }
}

fn render_test_console(out: &mut String, test_data: &TestReportData) {
    out.push_str(&format!("\n  {} — ", test_data.display_name));
    let primary_scores: Vec<_> = test_data
        .model_results
        .values()
        .filter_map(|br| br.scores.values().find(|s| s.primary))
        .collect();
    for score in primary_scores {
        out.push_str(&format!("{} ", score.display_value()));
    }
    out.push('\n');

    // Token usage
    for (model, br) in &test_data.model_results {
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
        out.push_str(&format!("    {}: {} / {}\n", model, output, thinking));
    }
}

impl ConsoleReport for ConsoleReportGenerator {}
