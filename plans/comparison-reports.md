# Comparison Reports Feature Plan

## Context

The benchmark runner generates a single monolithic `benchmark_report.html` file containing all models evaluated. When comparing specific models (e.g., "MyComparison" comparing Q4 vs Q5), users need a filtered view showing only those models in a dedicated report file, named after the comparison.

This plan adds a `comparisons` config section. Comparison reports are generated **automatically** alongside the main report when benchmarks run (and also when the `report` command is used). An additional `compare` CLI command provides the ability to regenerate only the comparison reports without re-running benchmarks.

## Approach

1. **Configuration** — Add a top-level `comparisons` key to the YAML config with a list of comparison entries, each having `title` and `models` (list of model display names).

2. **Automatic generation during normal execution** — The `run` command already calls `report::generate_reports` at the end of a benchmark run. We will extend this to also produce per-comparison reports by:
   - Passing the `comparisons` config list to the report generation phase.
   - After generating the main `benchmark_report.html`, iterating over each comparison, filtering the results JSON for that comparison's models, and writing a standalone HTML report with a slugified filename.
   - The `report` CLI command (regenerates from existing results) will also generate comparison reports when the config is provided.

3. **Optional standalone `compare` CLI command** — For convenience, a `compare` subcommand lets you regenerate only comparison reports from existing results without re-running benchmarks. It is not required for the primary workflow.

4. **Slugified filenames** — Generate the output filename by slugifying the comparison title: `my-comparison.html` from "MyComparison".

5. **Reuse existing report infrastructure** — The existing `generate_reports` function in `src/report.rs` generates both HTML and markdown from `results.json`. For comparison reports, we'll filter the results JSON to only include the specified models, then delegate to the same report generation but with a distinct filename.

## Config Scheme

Add to the YAML config:

```yaml
comparisons:
  - title: "Q4 vs Q5"
    models:
      - "MyModel Q4"
      - "MyModel Q5"
  - title: "All Models"
    models:
      - "MyModel Q4"
      - "MyModel Q5"
```

## Files to Modify

### 1. `src/config.rs`

- Add a new `Comparison` struct:
  ```rust
  pub struct Comparison {
      pub title: String,
      pub models: Vec<String>,
  }
  ```

- Add `comparisons: Vec<Comparison>` field to `Config` struct, with `#[serde(default)]` for backward compatibility.

- Optionally export a helper for slugification (reusable in CLI).

### 2. `src/main.rs`

- **Modify `run_benchmarks`**: After the existing call to `report::generate_reports`, pass the `config.comparisons` list. The report function will then also generate per-comparison reports.

- **Modify `generate_report`** (the `report` CLI command): Accept the config path as a parameter and pass comparisons to the report generator, so re-generated reports also include comparison HTML files.

- **Add `Compare` CLI subcommand** (optional, extra): Reads the config and results, then generates only comparison reports. This allows regenerating comparison reports without the main report.

- Implement `fn generate_comparison_reports(config_path: &str, results_path: &str, output_dir: &str) -> Result<()>` that:
  - Loads config to get `comparisons` list.
  - Loads results JSON.
  - For each comparison:
    - Filters the `models` map to only include keys in the `models` list.
    - Generates a slugified filename from the title (e.g., "Q4 vs Q5" → "q4-vs-q5.html").
    - Calls `report::generate_comparison_report` for the filtered results.
  - Handles edge cases: model names not found (skip model, warn), empty comparisons list (no reports generated).

- Wire up `compare` CLI command in `main()` match block.

### 3. `src/report.rs`

- Add a new public function `generate_comparison_report(
    results: &serde_json::Value,
    output_dir: &Path,
    filename: &str,
) -> Result<()>` that:
  - Takes pre-filtered results (the `models` object already contains only the desired models).
  - Delegates to the existing report generation logic (`ReportTemplate`) to produce HTML.
  - Writes the output to `output_dir/filename.html`.

- Consider also generating a matching `.md` file, unless the design decision is to produce only HTML for comparisons (since markdown output adds little value for filtered views).

### 4. `src/utils.rs` (new helper)

- Add a `slugify` function:
  ```rust
  pub fn slugify(title: &str) -> String {
      title
          .to_lowercase()
          .chars()
          .filter(|c| c.is_alphanumeric() || c == &' ')
          .map(|c| if c.is_whitespace() { '-' } else c })
          .collect::<String>()
          .split('-')
          .filter(|s| !s.is_empty())
          .collect::<Vec<_>>()
          .join("-")
  }
  ```

- This ensures "Q4 vs Q5" → "q4-vs-q5", "MyComparison" → "mycomparison", etc.

## Steps

- [ ] Add `slugify` utility to `src/utils.rs`.
- [ ] Add `Comparison` struct and `comparisons` field to `Config` in `src/config.rs`.
- [ ] Add `generate_comparison_report` function to `src/report.rs` that accepts filtered results and writes an HTML file.
- [ ] Update `generate_reports` in `src/report.rs` to accept an optional `comparisons` list and also generate per-comparison reports.
- [ ] Update `run_benchmarks` in `src/main.rs` to pass the `comparisons` config when calling `report::generate_reports`.
- [ ] Update the `report` CLI command to accept and pass comparisons.
- [ ] Add `Compare` CLI subcommand (optional) to `src/main.rs` for generating only comparison reports.
- [ ] Wire up the command in `main()` match block.
- [ ] Add example `comparisons` section to the sample `models_config.yaml` for documentation.
- [ ] Update `README.md` with a brief explanation of the new feature.

## Reuse

- **Existing `generate_reports`**: The core report logic (template rendering, data extraction) stays unchanged. The comparison report reuses the same `ReportTemplate` and all extraction functions — only the input data is filtered before passing it in.
- **`serde_json::Value` manipulation**: The existing code already uses `serde_json::Value` extensively for flexible result handling. The comparison filtering will simply construct a new JSON object with a filtered `models` map.
- **Askama templates**: No template changes needed — the same `report.html` template is used.
- **Result file format**: The `results.json` schema (`models` key with model-name keys) is reused directly.

## Verification

1. **Normal run**: Add a comparison to `models_config.yaml`, then run `cargo run -- run`. After the benchmark finishes, verify that in `benchmark_results/` there are both `benchmark_report.html` (main) and comparison files like `q4-vs-q5.html`.

2. **Filtered content**: Open the comparison HTML file and confirm it contains only the models listed in that comparison — no extra model rows in any table.

3. **Re-generate via report command**: Run `cargo run -- report -c models_config.yaml`. Verify comparison reports are also regenerated.

4. **Standalone compare command**: Run `cargo run -- compare -c models_config.yaml` and verify it produces the same comparison reports.

5. **Multiple comparisons**: Define 2–3 comparisons in the config; ensure each produces a distinct, correctly-named HTML file.

6. **Edge cases**:
   - Model name in a comparison that doesn't exist in results → report shows empty/no data, no crash.
   - Empty comparisons list → no comparison reports generated, no error.
   - Comparison with a single model → valid report with one row.

7. **Existing commands unaffected**: Ensure the `run` command without a `comparisons` section still generates the main report identically.

## Example Config Addition

```yaml
comparisons:
  - title: "Quantization Comparison"
    models:
      - "MyModel Q4"
      - "MyModel Q5"
  - title: "All Models"
    models:
      - "MyModel Q4"
      - "MyModel Q5"
```

This produces (automatically, after any `run` or `report` command):
- `benchmark_results/q4-vs-q5.html` (from slugifying "Quantization Comparison")
- `benchmark_results/all-models.html`

Each file is a standalone benchmark report containing only the models from that comparison, with all per-benchmark tables, summaries, and KLD sections correctly filtered.
