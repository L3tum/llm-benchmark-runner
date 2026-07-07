# Plan: Add Benchmark Categories to Result HTML

## Context
The user wants to add **categories** to group existing benchmarks and display them in the HTML report. This helps organize results by capability area (knowledge, math, coding, etc.) and makes the report easier to scan.

## Proposed Categories & Benchmark Mapping
| Category | Benchmarks |
|---|---|
| **Knowledge** | `mmlu_pro`, `gpqa` |
| **Math** | `aime`, `math500` |
| **Short-Context-Coding** | `coding_eval` (covers humaneval, humaneval_plus, mbpp_plus) |
| **Long-Context-Coding** | `swe_bench` (covers swebench, swebench_verified, swebench_pro) |
| **Creative** | `minebench` |
| **Reasoning** | *(empty for now — reserved for future benchmarks)* |
| **Research** | *(empty for now — reserved for future benchmarks)* |
| **Similarity** | `kld` (distribution similarity, lower = more similar) |

Additional categories I considered and why they're included:
- **Creative**: Minebench requires creative Minecraft world-building via JSON — it doesn't fit neatly into coding or reasoning, so this makes sense.
- **Similarity**: KLD is a meta-metric (comparing model distributions), not a capability test, so it deserves its own category.
- **Reasoning** & **Research**: Empty but reserved, as these are common future benchmark types (e.g., chain-of-thought reasoning benchmarks, SWE-bench could arguably be research/long-context).

## Files to Modify
1. **`src/report.rs`** — Add category mapping, category-aware report rendering, tab navigation
2. **`templates/report.html`** — Add tab navigation for categories, restructure to render per-category sections

## Approach
1. **Define a category map** (static HashMap) in `src/report.rs` mapping each benchmark key to its category (e.g. `"mmlu_pro" → "Knowledge"`).
2. **Extract per-category results** by grouping existing result data.
3. **Add tabbed navigation** in the HTML template (using JavaScript to switch between category tabs).
4. **Render benchmark tables within each category tab** — reuse existing table logic.
5. **Update `generate_reports`** and `render_report_html` to support category-aware rendering.
6. **Handle comparison reports** — the per-comparison filter should also work with categories.

## Reuse (existing functions/utilities)
- `extract_mmlu_results`, `extract_gpqa_results`, `extract_aime_results`, etc. — already parse benchmark results
- `generate_summary` — already collects top results per benchmark; can be extended with per-category summaries
- `pct()`, `format_optional_u64` — formatting helpers
- Template iteration patterns (`{% for entry in ... %}`) — already well-established

## Steps
- [ ] Add `BENCHMARK_CATEGORIES` static HashMap in `report.rs`
- [ ] Create a `CategoryResult` struct to hold all results for a given category
- [ ] Modify `render_report_html` to produce category-grouped data
- [ ] Add category tabbed navigation to `templates/report.html`
- [ ] Add JavaScript for tab switching (simple, inline in the template)
- [ ] Update `ReportTemplate` struct to include category data
- [ ] Extend `generate_summary` to optionally include per-category highlights
- [ ] Verify HTML report renders with tabs and correct per-category content
- [ ] Ensure comparison reports still work (filter models within each category)

## Verification
- Run benchmarks with existing configuration and check that the new HTML report shows tabs with the correct benchmarks grouped under each category.
- Open `benchmark_report.html` and confirm:
  - Tabs for Knowledge, Math, Short-Context-Coding, Long-Context-Coding, Creative, Reasoning, Research, Similarity appear
  - Each tab shows the relevant benchmark tables (e.g., Knowledge tab shows MMLU-Pro and GPQA)
  - Clicking a tab displays only that category's data
- Check comparison reports also render correctly with category grouping.
