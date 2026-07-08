use super::model::ReportInput;
use anyhow::Result;

/// Context passed to report generators.
pub struct ReportContext<'a> {
    pub input: &'a ReportInput,
}

/// Core report generator trait.
pub trait ReportGenerator {
    type Output;
    fn generate(&self, ctx: &ReportContext<'_>) -> Result<Self::Output>;
}

/// Marker trait for HTML report generators.
pub trait HtmlReport: ReportGenerator<Output = String> {}

/// Marker trait for Markdown report generators.
pub trait MarkdownReport: ReportGenerator<Output = String> {}

/// Marker trait for Console report generators.
pub trait ConsoleReport: ReportGenerator<Output = String> {}
