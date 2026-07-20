use crate::error_classes::WrongAnswerClass;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::BTreeMap;

/// Unique name for a benchmark (e.g., "mmlu_pro", "gpqa")
#[derive(Clone, Debug, Eq, PartialEq, Ord, PartialOrd, Hash, Serialize, Deserialize)]
pub struct TestName(pub String);

impl TestName {
    pub fn new(name: impl Into<String>) -> Self {
        Self(name.into())
    }
}

/// Top-level benchmark category for grouping and display.
#[derive(Clone, Debug, Eq, PartialEq, Ord, PartialOrd, Hash, Serialize, Deserialize)]
pub enum BenchmarkCategory {
    Knowledge,
    Math,
    ShortContextCoding,
    LongContextCoding,
    Creative,
    Reasoning,
    Research,
    Similarity,
    InstructionFollowing,
    Hallucination,
    Translation,
    Safety,
    Other(String),
}

impl BenchmarkCategory {
    pub fn display(&self) -> String {
        match self {
            Self::Knowledge => "Knowledge".to_string(),
            Self::Math => "Math".to_string(),
            Self::ShortContextCoding => "Short-Context-Coding".to_string(),
            Self::LongContextCoding => "Long-Context-Coding".to_string(),
            Self::Creative => "Creative".to_string(),
            Self::Reasoning => "Reasoning".to_string(),
            Self::Research => "Research".to_string(),
            Self::Similarity => "Similarity".to_string(),
            Self::InstructionFollowing => "Instruction-Following".to_string(),
            Self::Hallucination => "Hallucination".to_string(),
            Self::Translation => "Translation".to_string(),
            Self::Safety => "Safety".to_string(),
            Self::Other(s) => s.clone(),
        }
    }

    pub fn from_str(s: &str) -> Self {
        match s {
            "Knowledge" => Self::Knowledge,
            "Math" => Self::Math,
            "Short-Context-Coding" => Self::ShortContextCoding,
            "Long-Context-Coding" => Self::LongContextCoding,
            "Creative" => Self::Creative,
            "Reasoning" => Self::Reasoning,
            "Research" => Self::Research,
            "Similarity" => Self::Similarity,
            "Instruction-Following" => Self::InstructionFollowing,
            "Hallucination" => Self::Hallucination,
            "Translation" => Self::Translation,
            "Safety" => Self::Safety,
            other => Self::Other(other.to_string()),
        }
    }
}

/// Numeric or textual value for a score metric.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum ScoreValue {
    Float(f64),
    Integer(i64),
    Bool(bool),
    Text(String),
    Missing,
}

/// Unit of measurement for a score.
#[derive(Clone, Debug, Eq, PartialEq, Ord, PartialOrd, Serialize, Deserialize)]
pub enum ScoreUnit {
    Percent,
    Count,
    Tokens,
    Seconds,
    Ratio,
    Kld,
    Text,
    None,
}

impl ScoreUnit {
    /// Returns a human-readable unit string for display (e.g., "bits" for KLD).
    pub fn display(&self) -> &'static str {
        match self {
            Self::Percent => "%",
            Self::Count => "",
            Self::Tokens => "tokens",
            Self::Seconds => "s",
            Self::Ratio => "",
            Self::Kld => "bits",
            Self::Text => "",
            Self::None => "",
        }
    }
}

/// A single metric score, with formatting hints and directionality.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Score {
    pub value: ScoreValue,
    pub unit: ScoreUnit,
    pub display: Option<String>,
    pub higher_is_better: Option<bool>,
    pub primary: bool,
}

impl Score {
    pub fn float(value: f64, unit: ScoreUnit) -> Self {
        Self {
            value: ScoreValue::Float(value),
            unit,
            display: None,
            higher_is_better: None,
            primary: false,
        }
    }

    pub fn integer(value: i64, unit: ScoreUnit) -> Self {
        Self {
            value: ScoreValue::Integer(value),
            unit,
            display: None,
            higher_is_better: None,
            primary: false,
        }
    }

    pub fn text(value: String) -> Self {
        Self {
            value: ScoreValue::Text(value),
            unit: ScoreUnit::Text,
            display: None,
            higher_is_better: None,
            primary: false,
        }
    }

    pub fn missing() -> Self {
        Self {
            value: ScoreValue::Missing,
            unit: ScoreUnit::None,
            display: Some("–".into()),
            higher_is_better: None,
            primary: false,
        }
    }

    /// Create a boolean score (e.g., pass/fail).
    pub fn bool(value: bool) -> Self {
        Self {
            value: ScoreValue::Bool(value),
            unit: ScoreUnit::None,
            display: None,
            higher_is_better: None,
            primary: false,
        }
    }

    /// Mark this score as the primary metric.
    #[must_use]
    pub fn primary(mut self, v: bool) -> Self {
        self.primary = v;
        self
    }

    /// Indicate whether higher is better.
    #[must_use]
    pub fn higher_is_better(mut self, v: bool) -> Self {
        self.higher_is_better = Some(v);
        self
    }

    /// Set a pre-formatted display string.
    #[must_use]
    pub fn display(mut self, s: String) -> Self {
        self.display = Some(s);
        self
    }

    /// Display value as percent if primary, otherwise as raw, appending units when appropriate.
    pub fn display_value(&self) -> String {
        if let Some(ref d) = self.display {
            d.clone()
        } else {
            let val_str = self.value.display();
            // Append unit indicator if it's not percent (which already has %) and not a text/count type
            match &self.unit {
                ScoreUnit::Kld => format!("{} bits", val_str),
                ScoreUnit::Tokens => format!("{} tokens", val_str),
                _ => val_str,
            }
        }
    }
}

impl ScoreValue {
    pub fn display(&self) -> String {
        match self {
            ScoreValue::Float(v) => format!("{:.1}", v),
            ScoreValue::Integer(v) => v.to_string(),
            ScoreValue::Bool(v) => {
                if *v {
                    "✓".to_string()
                } else {
                    "✗".to_string()
                }
            }
            ScoreValue::Text(t) => t.clone(),
            ScoreValue::Missing => "–".to_string(),
        }
    }
}

// Manual PartialEq/PartialOrd because f64 doesn't implement Eq/Ord
impl PartialEq for ScoreValue {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (Self::Float(a), Self::Float(b)) => a == b,
            (Self::Integer(a), Self::Integer(b)) => a == b,
            (Self::Bool(a), Self::Bool(b)) => a == b,
            (Self::Text(a), Self::Text(b)) => a == b,
            (Self::Missing, Self::Missing) => true,
            _ => false,
        }
    }
}

impl PartialOrd for ScoreValue {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        match (self, other) {
            (Self::Float(a), Self::Float(b)) => a.partial_cmp(b),
            (Self::Integer(a), Self::Integer(b)) => a.partial_cmp(b),
            (Self::Bool(a), Self::Bool(b)) => a.partial_cmp(b),
            (Self::Text(a), Self::Text(b)) => a.partial_cmp(b),
            (Self::Missing, Self::Missing) => Some(std::cmp::Ordering::Equal),
            _ => None,
        }
    }
}

/// A breakdown table (e.g., per-subject accuracy).
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct BreakdownTable {
    pub title: String,
    pub rows: BTreeMap<String, BTreeMap<String, Score>>,
}

/// An artifact associated with a benchmark run (file, URL, etc.).
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Artifact {
    pub label: String,
    pub path: String,
    pub kind: String,
}

/// A diagnostic message (warning, error).
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Diagnostic {
    pub level: String,
    pub message: String,
}

/// Normalized benchmark result with generic scores and optional breakdowns.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct BenchmarkResult {
    /// Generic metrics usable by any report generator.
    pub scores: BTreeMap<String, Score>,
    /// Optional structured benchmark-specific tables.
    pub breakdowns: BTreeMap<String, BreakdownTable>,
    /// Per-error-class breakdown keyed by the enum (used by all benchmarks for error classification).
    pub error_classification: BTreeMap<WrongAnswerClass, i64>,
    pub artifacts: Vec<Artifact>,
    pub diagnostics: Vec<Diagnostic>,
    /// Raw benchmark payload for backwards compatibility and custom renderers.
    pub raw: Value,
}

impl BenchmarkResult {
    pub fn empty() -> Self {
        Self {
            scores: BTreeMap::new(),
            breakdowns: BTreeMap::new(),
            error_classification: BTreeMap::new(),
            artifacts: vec![],
            diagnostics: vec![],
            raw: Value::Null,
        }
    }
}

/// Aggregate per-test results (pairwise comparisons, global summaries).
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct TestAggregate {
    pub scores: BTreeMap<String, Score>,
    pub breakdowns: BTreeMap<String, BreakdownTable>,
    pub raw: Value,
}

/// Full report data for a single benchmark.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct TestReportData {
    pub name: TestName,
    pub display_name: String,
    pub category: BenchmarkCategory,
    pub model_results: BTreeMap<String, BenchmarkResult>,
    pub aggregate: Option<TestAggregate>,
}

/// Top-level report input containing all benchmark results.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ReportInput {
    pub generated_at: String,
    pub models: Vec<String>,
    pub tests: BTreeMap<TestName, TestReportData>,
    pub summary: Vec<String>,
    /// Raw JSON for backwards compatibility and custom renderers.
    pub raw_results: Value,
}
