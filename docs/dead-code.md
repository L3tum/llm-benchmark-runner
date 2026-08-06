# Dead Code Audit

Audit date: 2026-08-05

This document catalogues and justifies every `#[allow(dead_code)]` suppression in
the codebase. The audit concluded that **all 50 suppressions are intentional and
safe** — none masks an active bug. Each falls into one of five categories below.

When working in a file that uses `#[allow(dead_code)]`, prefer to:

- Reuse the existing field/function rather than adding a parallel one.
- Remove the suppression only if you are certain the symbol is now genuinely
  reachable (the compiler only reports the *first* definition as dead).

---

## 1. Schema-alignment fields (dataset deserialization) — most common

Many benchmarks deserialize datasets from HuggingFace / locally-cached JSON. The
source JSON contains more fields than the benchmark consumes; the extra struct
fields are kept so the `Deserialize` schema stays aligned with the source and so
report round-trips don't drop data.

| File:line | Field | Purpose |
|-----------|-------|---------|
| `benchmarks/scifact.rs:55` | `extra` (`#[serde(flatten)]`) | Tolerant catch-all for unused dataset fields |
| `benchmarks/triviaqa.rs:33` | `question_source` | Schema alignment |
| `benchmarks/tool_hallucination.rs:37,42` | `details` | Schema alignment |
| `benchmarks/mmlu_prox.rs:33` | `subject` | Schema alignment |
| `benchmarks/xsum.rs:40` | `narrative_link` | Schema alignment |
| `benchmarks/stable_toolbench.rs:26` | `query_id` | Schema alignment |
| `benchmarks/stable_toolbench.rs:64` | `default` | Schema alignment |
| `benchmarks/ea_mt.rs:33,44` | `entities` / entity fields | Schema alignment |
| `benchmarks/cnn_dailymail.rs:38` | `id` | Schema alignment |
| `benchmarks/squad_v2.rs:33,43` | `title`, `answer_start` | Schema alignment |
| `benchmarks/mmlu_pro_plus.rs:34` | `id`, `category` | Schema alignment |
| `benchmarks/terminal_bench.rs:82,87,90` | several | Kept for schema completeness |
| `shared.rs:80` | `BenchmarkCategory::from_display_name` | Report deserialization round-trip utility |

## 2. Reserved-for-future features / filtering

Fields reserved so future features (multi-language evaluation, dataset filtering,
category aggregation, richer reporting) can be enabled without re-adding schema.

| File:line | Field | Reserved for |
|-----------|-------|--------------|
| `benchmarks/coding_eval.rs:292,301` | config fields | Future multi-language + full harness support |
| `benchmarks/terminal_bench.rs:105,107,111` | `num_samples`, `categories` | Future filtering/sampling |
| `benchmarks/stable_toolbench.rs:77` | `num_samples`, `subsets`, `categories` | Future filtering |
| `benchmarks/stable_toolbench.rs:565` | reporting fields | Future detailed reporting |
| `benchmarks/swe_bench.rs:649,651,653` | diagnostics | Captured for future diagnostics |
| `benchmarks/mmlu_pro.rs:103` | axis/category field | Category-level aggregation |

## 3. Debugging / testing helpers

Helpers kept because they are useful for debugging a single case or inspecting
harness output — not referenced on the hot execution path.

| File:line | Symbol | Purpose |
|-----------|--------|---------|
| `benchmarks/multipl_e.rs:524` | `evaluate_in_docker` | Debug single-instance eval (batch path uses `evaluate_batch_in_docker`) |
| `benchmarks/swe_bench.rs:1601,1636` | harness-output helpers | Future debugging of harness output |
| `benchmarks/swe_bench.rs:1311` | `build_patch_prompt` | Kept for harness/prediction prompt generation (not on hot path) |

## 4. Consumed by non-obvious code paths

These look dead to the compiler but are actually read elsewhere (e.g., the agent
loop in `execute_one`, or the harness prediction format).

| File:line | Symbol | Actually consumed by |
|-----------|--------|---------------------|
| `benchmarks/swe_bench.rs:198,201` | config fields | `execute_one` agent loop |
| `benchmarks/swe_bench.rs:231` | harness fields | Prediction-format harness |

## 5. Migration utilities

| File:line | Symbol | Purpose |
|-----------|--------|---------|
| `download.rs:294,408` | arrow/parquet converters (`arrow_value_to_json`, etc.) | Used by benchmarks during the HF-parquet dataset migration |
| `docker_runner.rs:15,34` | mount-config fields | Reserved for mount configuration |

---

## Conclusion

All 50 `#[allow(dead_code)]` suppressions were reviewed; every one is justified
(dataset schema alignment, reserved future features, debugging helpers, or
consumed-by-portal code paths). **No suppression masks a bug, and none should be
removed without first confirming the surrounding feature is implemented.** The
codebase deliberately keeps dataset fields aligned with source JSON for tolerant,
lossless deserialization.
