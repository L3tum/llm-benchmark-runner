use super::model::BenchmarkCategory;
use std::sync::OnceLock;

/// Ordered list of category names for display (includes reserved empty categories).
static CATEGORY_ORDER: OnceLock<Vec<BenchmarkCategory>> = OnceLock::new();

pub fn get_category_order() -> &'static Vec<BenchmarkCategory> {
    CATEGORY_ORDER.get_or_init(|| {
        vec![
            BenchmarkCategory::Knowledge,
            BenchmarkCategory::Math,
            BenchmarkCategory::ShortContextCoding,
            BenchmarkCategory::LongContextCoding,
            BenchmarkCategory::Creative,
            BenchmarkCategory::Reasoning,
            BenchmarkCategory::Research,
            BenchmarkCategory::Similarity,
            BenchmarkCategory::Hallucination,
            BenchmarkCategory::Translation,
        ]
    })
}

/// Slugify a category name for HTML tab IDs.
pub fn slugify_name(name: String) -> String {
    name.to_lowercase()
        .replace(" ", "-")
        .replace("/", "-")
        .replace("_", "-")
        .replace("  ", "-")
        .trim()
        .to_string()
}
