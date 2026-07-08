# Plan: report generator traits + normalized benchmark results

## Verification

This design should work, with one important Rust architecture constraint:

- A real subcrate can live at `src/reports` if the root `Cargo.toml` adds it as a path dependency, e.g. `reports = { path = "src/reports" }`.
- That subcrate cannot depend on the parent binary crate without a circular dependency. Therefore, the shared report input/result model must be owned by the `reports` crate, or by another shared crate. For this repo, owning those types in `src/reports` is the simplest path.
- The parent crate can depend on `reports`, benchmark implementations can construct `reports::model::*` types, and `reports` can contain HTML/Markdown/Console implementations without importing app config/runner code.

The normalized shape `HashMap<TestName, HashMap<String, Result>>` is viable, but KLD/pairwise/post-execute results are not purely per-model. The data model should include both per-model test results and optional aggregate/per-test results.

## Target architecture

```text
root crate
├── src/main.rs
├── src/runner.rs
├── src/benchmarks/*        # benchmark execution, constructs normalized result data
├── src/report.rs           # thin orchestration / compatibility wrapper initially
└── src/reports/            # path dependency crate named `reports`
    ├── Cargo.toml
    ├── src/lib.rs
    ├── src/model.rs        # TestName, BenchmarkResult, Score, ReportInput, schemas
    ├── src/generator.rs    # report generator traits
    ├── src/html.rs         # HtmlReportGenerator
    ├── src/markdown.rs     # MarkdownReportGenerator
    ├── src/console.rs      # ConsoleReportGenerator
    ├── src/categories.rs   # category mapping/order
    └── templates/report.html
```

## Core normalized data model

Use ordered maps where stable report output is desirable.

```rust
pub type ModelName = String;
pub type MetricName = String;
pub type SectionName = String;
pub type RowName = String;

#[derive(Clone, Debug, Eq, PartialEq, Ord, PartialOrd, Hash, Serialize, Deserialize)]
pub struct TestName(pub String);

#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum BenchmarkCategory {
    Knowledge,
    Math,
    ShortContextCoding,
    LongContextCoding,
    Creative,
    Reasoning,
    Research,
    Similarity,
    Other(String),
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum ScoreValue {
    Float(f64),
    Integer(i64),
    Bool(bool),
    Text(String),
    Missing,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
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

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Score {
    pub value: ScoreValue,
    pub unit: ScoreUnit,
    pub display: Option<String>,
    pub higher_is_better: Option<bool>,
    pub primary: bool,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct BreakdownTable {
    pub title: String,
    pub rows: BTreeMap<RowName, BTreeMap<MetricName, Score>>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Artifact {
    pub label: String,
    pub path: String,
    pub kind: String,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Diagnostic {
    pub level: String,
    pub message: String,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct BenchmarkResult {
    /// Generic metrics usable by every report generator.
    pub scores: BTreeMap<MetricName, Score>,

    /// Optional structured benchmark-specific tables, e.g. subjects, tasksets, failures.
    pub breakdowns: BTreeMap<SectionName, BreakdownTable>,

    pub artifacts: Vec<Artifact>,
    pub diagnostics: Vec<Diagnostic>,

    /// Raw benchmark payload retained for backwards compatibility and custom renderers.
    pub raw: serde_json::Value,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct TestAggregate {
    /// For KLD pairwise, global tables, cross-model summaries, etc.
    pub scores: BTreeMap<MetricName, Score>,
    pub breakdowns: BTreeMap<SectionName, BreakdownTable>,
    pub raw: serde_json::Value,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct TestReportData {
    pub name: TestName,
    pub display_name: String,
    pub category: BenchmarkCategory,
    pub model_results: BTreeMap<ModelName, BenchmarkResult>,
    pub aggregate: Option<TestAggregate>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ReportInput {
    pub generated_at: String,
    pub models: Vec<ModelName>,
    pub tests: BTreeMap<TestName, TestReportData>,
    pub summary: Vec<String>,
    pub raw_results: serde_json::Value,
}
```

This preserves the requested shape via `ReportInput.tests[test_name].model_results[model_name]`, while also handling aggregate results.

## Benchmark trait changes

Current trait returns `serde_json::Value`:

```rust
fn execute(&self, model: &Model, config: &serde_yaml::Value) -> Result<serde_json::Value>;
fn post_execute(&self, model_results: &HashMap<String, serde_json::Value>) -> Result<serde_json::Value>;
```

Target trait should add metadata first, then eventually return normalized data:

```rust
pub trait Benchmark: Send + Sync {
    fn name(&self) -> &str;
    fn test_name(&self) -> reports::model::TestName {
        reports::model::TestName(self.name().to_string())
    }
    fn display_name(&self) -> &'static str;
    fn category(&self) -> reports::model::BenchmarkCategory;

    // Phase 1 compatibility: keep existing JSON output.
    fn execute(&self, model: &Model, config: &serde_yaml::Value) -> Result<serde_json::Value>;
    fn post_execute(&self, model_results: &HashMap<String, serde_json::Value>) -> Result<serde_json::Value>;

    // Phase 2/3 target: benchmark-specific adapter into generic report data.
    fn to_report_result(&self, raw: &serde_json::Value) -> Result<reports::model::BenchmarkResult>;
    fn to_report_aggregate(&self, raw: &serde_json::Value) -> Result<Option<reports::model::TestAggregate>> {
        Ok(None)
    }
}
```

Later, after compatibility is stable, `execute` can return `BenchmarkResult` directly and raw JSON can be derived from it.

## Report generator traits

Use one common trait for all generators, plus marker/extension traits for HTML/Markdown/Console.

```rust
pub struct ReportContext<'a> {
    pub input: &'a ReportInput,
}

pub trait ReportGenerator {
    type Output;
    fn generate(&self, ctx: &ReportContext<'_>) -> anyhow::Result<Self::Output>;
}

pub trait HtmlReport: ReportGenerator<Output = String> {}
pub trait MarkdownReport: ReportGenerator<Output = String> {}
pub trait ConsoleReport: ReportGenerator<Output = String> {}
```

Implementations:

```rust
pub struct HtmlReportGenerator;
pub struct MarkdownReportGenerator;
pub struct ConsoleReportGenerator;

impl ReportGenerator for HtmlReportGenerator { type Output = String; /* Askama render */ }
impl HtmlReport for HtmlReportGenerator {}

impl ReportGenerator for MarkdownReportGenerator { type Output = String; /* markdown text */ }
impl MarkdownReport for MarkdownReportGenerator {}

impl ReportGenerator for ConsoleReportGenerator { type Output = String; /* compact terminal text */ }
impl ConsoleReport for ConsoleReportGenerator {}
```

The root app remains responsible for writing files/printing:

```rust
let ctx = ReportContext { input: &report_input };
let html = HtmlReportGenerator.generate(&ctx)?;
fs::write(output_dir.join("benchmark_report.html"), html)?;

let md = MarkdownReportGenerator.generate(&ctx)?;
fs::write(output_dir.join("benchmark_report.md"), md)?;

let console = ConsoleReportGenerator.generate(&ctx)?;
println!("{console}");
```

## Generic report behavior

Every report generator should support generic rendering from `BenchmarkResult.scores` and `breakdowns`:

- Primary score columns first (`Score.primary == true`).
- Common token metrics next (`output_tokens`, `thinking_tokens`) if present.
- Other metrics sorted by metric name.
- Breakdown tables rendered below the primary table.
- Diagnostics rendered as warnings/errors.
- Artifacts rendered as links or paths.

Benchmark-specific rendering is optional and additive:

```rust
pub trait BenchmarkReportRenderer {
    fn test_name(&self) -> &TestName;
    fn render_html(&self, data: &TestReportData) -> Option<anyhow::Result<String>>;
    fn render_markdown(&self, data: &TestReportData) -> Option<anyhow::Result<String>>;
}
```

If no benchmark-specific renderer exists, use the generic renderer. This makes new benchmarks reportable immediately once they provide scores.

## Mapping existing benchmarks to generic scores

Initial required fields per benchmark:

### MMLU-Pro / GPQA / MATH-500

Scores:
- `accuracy`: percent, primary, higher is better
- `total_questions`: count
- `output_tokens`: tokens
- `thinking_tokens`: tokens

Breakdowns:
- `subjects`: row = subject/category, metrics = `accuracy`, `correct`, `wrong`

### AIME

Scores:
- `accuracy`: percent, primary, higher is better
- `correct`: count
- `total_questions`: count
- `output_tokens`: tokens
- `thinking_tokens`: tokens

### Coding Eval / HumanEval / MBPP

Scores:
- `pass_at_1`: percent, primary, higher is better
- `pass_at_2`: percent if present
- `pass_at_3`: percent if present
- `passed`: count
- `total_questions`: count
- `timeout_count`: count, lower is better
- `skipped_later_attempts`: count, lower is better
- `output_tokens`: tokens
- `thinking_tokens`: tokens

Breakdowns:
- `tasksets`: row = taskset, metrics = pass@N, passed, total, timeout_count, skipped_later_attempts
- `failures`: row = task id, metrics/text = taskset, entry_point, error_summary

### SWE-Bench

Scores:
- `resolution_rate`: percent, primary, higher is better
- `resolved`: count
- `total_questions`: count
- `harness_passed`: bool
- `output_tokens`: tokens
- `thinking_tokens`: tokens
- `error_summary`: text

### Minebench

Scores:
- `json_valid`: bool, primary
- `valid_buildings`: count
- `total_buildings`: count
- `output_file`: text/artifact
- `output_tokens`: tokens
- `thinking_tokens`: tokens

### KLD

Per-model average scores:
- `avg_kld_to_others`: KLD, primary, lower is better
- `output_tokens`: tokens
- `thinking_tokens`: tokens

Aggregate breakdown:
- `pairwise_kld`: row = `model_a | model_b`, metrics = avg_kld, num_prompts_evaluated

## Migration phases

### Phase 1: Extract report generators without changing result JSON

1. Add `src/reports/Cargo.toml` path crate.
2. Add `reports` dependency in root `Cargo.toml`.
3. Move category helpers and report data DTOs into `reports::model`.
4. Add `ReportGenerator`, `HtmlReport`, `MarkdownReport`, `ConsoleReport` traits.
5. Move current HTML template and renderer into `reports::html`.
6. Move current markdown rendering into `reports::markdown`.
7. Add a simple console renderer based on summary + primary scores.
8. Keep `src/report.rs` as orchestration/compatibility: extract current JSON into report DTOs, call generators, write files.
9. Verify `cargo fmt`, `cargo clippy --all-targets --all-features -- -D warnings`, and generated HTML/Markdown match current content.

### Phase 2: Introduce normalized `ReportInput`

1. Implement `ReportInput::from_legacy_json(results: &serde_json::Value)`.
2. Convert current `extract_*` functions into legacy adapters that output `TestReportData` / `BenchmarkResult`.
3. Update HTML/Markdown/Console generators to consume only `ReportInput`.
4. Keep writing existing `results.json` format to avoid breaking resume/report commands.
5. Comparison reports filter `ReportInput` by model names instead of filtering raw JSON where possible; keep raw JSON filter as fallback.

### Phase 3: Move normalization into benchmarks

1. Add `test_name`, `display_name`, `category`, and `to_report_result` to each benchmark.
2. Update runner to build normalized results alongside legacy JSON.
3. Add snapshot tests comparing normalized adapters against legacy extraction for each benchmark.
4. Once stable, decide whether `results.json` should remain legacy, switch to normalized, or include both:

```json
{
  "schema_version": 2,
  "models": { "legacy": "..." },
  "report": { "tests": "normalized ReportInput-ish data" }
}
```

### Phase 4: Remove benchmark-specific report coupling

1. Remove benchmark-specific typed structs from old `src/report.rs` once all generators use `ReportInput`.
2. Keep optional benchmark-specific renderers only where generic output is insufficient.
3. Document the minimum requirements for adding a new benchmark:
   - `name/test_name`
   - `category`
   - generic `scores`
   - optional `breakdowns`
   - optional custom renderer

## Implementation details / pitfalls

- Prefer `BTreeMap` over `HashMap` inside reports for deterministic output and snapshot tests.
- Keep `serde_json::Value raw` in `BenchmarkResult` during migration so no benchmark-specific detail is lost.
- Avoid trait objects for serialized results. Data should be concrete structs/enums; traits should generate/render from those structs.
- Use `Score.display` for pre-rounded strings like `"83.33"` so renderers do not duplicate rounding logic.
- Use `higher_is_better` to calculate best rows generically.
- Use `primary` to decide which score drives summary/highlighting.
- Keep report generators side-effect free: return strings. File writing and stdout printing should stay in the root crate.
- Askama templates in the `reports` crate should live at `src/reports/templates/report.html` unless configured otherwise.
- If `reports` is a path dependency under `src/reports`, add it to workspace/package metadata carefully and ensure clippy runs include it.

## Acceptance criteria

- Existing `Run`, `Report`, and `Compare` commands still work.
- `benchmark_report.html` displays all existing benchmark sections, now via the report generator trait.
- `benchmark_report.md` displays equivalent data via the markdown generator trait.
- Console report can be generated from the same `ReportInput` without accessing raw JSON directly.
- Adding a new benchmark with only generic scores produces useful HTML/Markdown/Console output without template changes.
- Existing legacy `results.json` can still be read by `Report`.
- Clippy passes with `-D warnings`.
