//! Report-specific data types.
//!
//! Shared types (BenchmarkResult, TaskResult, Score, etc.) are defined in
//! `crate::shared` and re-exported here for backward compatibility.
//! New code should import from `crate::shared` directly.

use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::BTreeMap;

// Backward-compatible re-exports — prefer `crate::shared::*` for new code.
pub use crate::shared::{
    Artifact, BenchmarkCategory, BenchmarkResult, BreakdownTable, Diagnostic, Score, ScoreUnit,
    ScoreValue, TaskResult, TestAggregate, TestName,
};

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
