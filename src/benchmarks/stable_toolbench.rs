use crate::benchmarks::Benchmark;
use crate::config;
use crate::config::Model;
use crate::shared::{
    BenchmarkCategory, BenchmarkResult, BreakdownTable, Score, ScoreUnit, TaskResult,
};
use crate::token_tracker::TokenTracker;
use anyhow::{Context, Result};
use serde::Deserialize;
use std::collections::BTreeMap;
use std::fs;
use std::path::PathBuf;
use std::sync::Mutex;

pub struct StableToolBenchBenchmark {
    state: Mutex<StableToolBenchState>,
}

struct StableToolBenchState {
    instances: Vec<SolvableQuery>,
    current_idx: usize,
    config: StableToolBenchConfig,
}

#[derive(Debug, Clone, Deserialize)]
#[allow(dead_code)] // query_id kept for schema alignment
struct SolvableQuery {
    #[serde(rename = "api_list")]
    api_list: Vec<ToolDefinition>,
    query: String,
    #[serde(rename = "query_id")]
    query_id: u64,
    #[serde(rename = "relevant APIs")]
    relevant_apis: Vec<Vec<String>>, // [[tool_name, api_name], ...]
    #[serde(default)]
    domain: Option<String>,
    /// Which subset this instance belongs to (populated at load time).
    #[serde(skip_deserializing, default)]
    subset: Option<String>,
}

#[derive(Debug, Clone, Deserialize)]
#[allow(dead_code)] // category_name, method, template_response kept for schema alignment
struct ToolDefinition {
    #[serde(default)]
    category_name: String,
    #[serde(default)]
    tool_name: String,
    #[serde(default)]
    api_name: String,
    #[serde(default)]
    api_description: String,
    #[serde(default, rename = "required_parameters")]
    required_parameters: Vec<ParamDef>,
    #[serde(default, rename = "optional_parameters")]
    optional_parameters: Vec<ParamDef>,
    #[serde(default)]
    method: String,
    #[serde(default, rename = "template response")]
    template_response: Option<serde_json::Value>,
}

#[derive(Debug, Clone, Deserialize)]
#[allow(dead_code)] // default field kept for schema alignment
struct ParamDef {
    #[serde(default)]
    name: String,
    #[serde(default, rename = "_type")]
    param_type: String,
    #[serde(default)]
    description: String,
    #[serde(default)]
    default: String,
}

#[derive(Debug, Clone)]
#[allow(dead_code)] // num_samples, subsets, categories reserved for future filtering
struct StableToolBenchConfig {
    num_samples: Option<usize>,
    subsets: Option<Vec<String>>,
    categories: Option<Vec<String>>,
}

impl Default for StableToolBenchBenchmark {
    fn default() -> Self {
        Self {
            state: Mutex::new(StableToolBenchState {
                instances: Vec::new(),
                current_idx: 0,
                config: StableToolBenchConfig {
                    num_samples: None,
                    subsets: None,
                    categories: None,
                },
            }),
        }
    }
}

impl StableToolBenchBenchmark {
    pub fn new() -> Self {
        Self::default()
    }
}

impl Benchmark for StableToolBenchBenchmark {
    fn name(&self) -> &str {
        "stable_toolbench"
    }

    fn display_name(&self) -> &'static str {
        "StableToolBench"
    }

    fn category(&self) -> BenchmarkCategory {
        BenchmarkCategory::ToolUse
    }

    fn pre_execute(&self, config: &yaml_serde::Value) -> Result<()> {
        let num_samples = config::extract_usize(config, "num_samples");
        let subsets = config::extract_string_vec(config, "subsets");
        let categories = config::extract_string_vec(config, "categories");

        // Download dataset
        let instances = download_dataset()?;

        let mut state = self.state.lock().expect(crate::shared::MUTEX_PANIC_MSG);
        state.instances = filter_instances(instances, &subsets, &categories, num_samples);
        state.config = StableToolBenchConfig {
            num_samples,
            subsets,
            categories,
        };
        state.current_idx = 0;

        println!(
            "  StableToolBench: {} instances loaded",
            state.instances.len()
        );
        Ok(())
    }

    fn execute_one(
        &self,
        model: &Model,
        _config: &yaml_serde::Value,
        tracker: &mut TokenTracker,
    ) -> Result<Option<TaskResult>> {
        let (instance, idx) = {
            let mut state = self.state.lock().expect(crate::shared::MUTEX_PANIC_MSG);
            if state.current_idx >= state.instances.len() {
                return Ok(None);
            }
            let idx = state.current_idx;
            let instance = state.instances[idx].clone();
            state.current_idx += 1;
            (instance, idx)
        };

        let (result, _) = evaluate_instance(&instance, idx, model, tracker)?;
        // Token counts populated by runner.rs from tracker delta
        Ok(Some(result))
    }

    fn to_report_result(&self, b: &BenchmarkResult) -> Result<BenchmarkResult> {
        let raw = &b.raw;

        // Both execute() and execute_one() paths now emit per_task format.
        // Extract metrics from the per_task array and the aggregate raw fields.
        let (
            pass_rate,
            tool_selection_accuracy,
            precision,
            recall,
            f1,
            param_completeness,
            total_instances,
            passed_instances,
            output_tokens,
            thinking_tokens,
            per_task_array,
        ) = {
            let per_task = raw.get("per_task").and_then(|v| v.as_array());

            if let Some(per_task) = per_task {
                let total = per_task.len() as i64;
                let passed = per_task
                    .iter()
                    .filter(|t| t.get("passed").and_then(|v| v.as_bool()).unwrap_or(false))
                    .count() as i64;
                let total_out: i64 = per_task
                    .iter()
                    .filter_map(|t| t.get("output_tokens").and_then(|v| v.as_i64()))
                    .sum();
                let total_think: i64 = per_task
                    .iter()
                    .filter_map(|t| t.get("thinking_tokens").and_then(|v| v.as_i64()))
                    .sum();

                let mut total_tp = 0usize;
                let mut total_fp = 0usize;
                let mut total_fn = 0usize;
                let mut tool_correct = 0usize;
                let mut param_complete_count = 0usize;

                for task in per_task {
                    if let Some(meta) = task.get("metadata").and_then(|v| v.as_object()) {
                        total_tp += meta
                            .get("api_true_positives")
                            .and_then(|v| v.as_u64())
                            .unwrap_or(0) as usize;
                        total_fp += meta
                            .get("api_false_positives")
                            .and_then(|v| v.as_u64())
                            .unwrap_or(0) as usize;
                        total_fn += meta
                            .get("api_false_negatives")
                            .and_then(|v| v.as_u64())
                            .unwrap_or(0) as usize;
                        if meta
                            .get("tool_selection_correct")
                            .and_then(|v| v.as_bool())
                            .unwrap_or(false)
                        {
                            tool_correct += 1;
                        }
                        if meta
                            .get("param_complete")
                            .and_then(|v| v.as_bool())
                            .unwrap_or(false)
                        {
                            param_complete_count += 1;
                        }
                    }
                }

                let p = if total_tp + total_fp > 0 {
                    total_tp as f64 / (total_tp + total_fp) as f64
                } else {
                    0.0
                };
                let r = if total_tp + total_fn > 0 {
                    total_tp as f64 / (total_tp + total_fn) as f64
                } else {
                    0.0
                };
                let f = if p + r > 0.0 {
                    2.0 * p * r / (p + r)
                } else {
                    0.0
                };

                (
                    if total > 0 {
                        passed as f64 / total as f64
                    } else {
                        0.0
                    },
                    if total > 0 {
                        tool_correct as f64 / total as f64
                    } else {
                        0.0
                    },
                    p,
                    r,
                    f,
                    if total > 0 {
                        param_complete_count as f64 / total as f64
                    } else {
                        0.0
                    },
                    total,
                    passed,
                    total_out,
                    total_think,
                    Some(per_task),
                )
            } else {
                // Fallback: read aggregate fields directly from raw (e.g., deserialized results)
                (
                    raw.get("pass_rate").and_then(|v| v.as_f64()).unwrap_or(0.0),
                    raw.get("tool_selection_accuracy")
                        .and_then(|v| v.as_f64())
                        .unwrap_or(0.0),
                    raw.get("api_precision")
                        .and_then(|v| v.as_f64())
                        .unwrap_or(0.0),
                    raw.get("api_recall")
                        .and_then(|v| v.as_f64())
                        .unwrap_or(0.0),
                    raw.get("api_f1").and_then(|v| v.as_f64()).unwrap_or(0.0),
                    raw.get("param_completeness")
                        .and_then(|v| v.as_f64())
                        .unwrap_or(0.0),
                    raw.get("total_instances")
                        .and_then(|v| v.as_i64())
                        .unwrap_or(0),
                    raw.get("passed_instances")
                        .and_then(|v| v.as_i64())
                        .unwrap_or(0),
                    raw.get("output_tokens")
                        .and_then(|v| v.as_i64())
                        .unwrap_or(0),
                    raw.get("thinking_tokens")
                        .and_then(|v| v.as_i64())
                        .unwrap_or(0),
                    None,
                )
            }
        };

        let mut scores = BTreeMap::new();
        scores.insert(
            "simulated_pass_rate".to_string(),
            Score::float(pass_rate, ScoreUnit::Percent)
                .primary(true)
                .higher_is_better(true)
                .display(format!("{:.1}%", pass_rate * 100.0)),
        );
        scores.insert(
            "tool_selection_accuracy".to_string(),
            Score::float(tool_selection_accuracy, ScoreUnit::Percent)
                .higher_is_better(true)
                .display(format!("{:.1}%", tool_selection_accuracy * 100.0)),
        );
        scores.insert(
            "api_precision".to_string(),
            Score::float(precision, ScoreUnit::Percent)
                .higher_is_better(true)
                .display(format!("{:.1}%", precision * 100.0)),
        );
        scores.insert(
            "api_recall".to_string(),
            Score::float(recall, ScoreUnit::Percent)
                .higher_is_better(true)
                .display(format!("{:.1}%", recall * 100.0)),
        );
        scores.insert(
            "api_f1".to_string(),
            Score::float(f1, ScoreUnit::Percent)
                .higher_is_better(true)
                .display(format!("{:.1}%", f1 * 100.0)),
        );
        scores.insert(
            "param_completeness".to_string(),
            Score::float(param_completeness, ScoreUnit::Percent)
                .higher_is_better(true)
                .display(format!("{:.1}%", param_completeness * 100.0)),
        );
        scores.insert(
            "instances_passed".to_string(),
            Score::integer(passed_instances, ScoreUnit::Count)
                .display(format!("{}/{}", passed_instances, total_instances)),
        );

        if output_tokens > 0 {
            scores.insert(
                "output_tokens".to_string(),
                Score::integer(output_tokens, ScoreUnit::Tokens),
            );
        }
        if thinking_tokens > 0 {
            scores.insert(
                "thinking_tokens".to_string(),
                Score::integer(thinking_tokens, ScoreUnit::Tokens),
            );
        }

        // Domain breakdown
        let mut breakdowns = BTreeMap::new();
        if let Some(per_task) = per_task_array {
            let mut domain_counts: BTreeMap<String, (i64, i64)> = BTreeMap::new();

            for task in per_task {
                // Try metadata.domain first (execute_one path), then domain directly (execute path)
                let domain = task
                    .get("metadata")
                    .and_then(|m| m.get("domain").and_then(|v| v.as_str()))
                    .or_else(|| task.get("domain").and_then(|v| v.as_str()))
                    .unwrap_or("Unknown")
                    .to_string();
                let passed = task
                    .get("passed")
                    .and_then(|v| v.as_bool())
                    .unwrap_or(false);

                let (p, t) = domain_counts.entry(domain).or_insert((0, 0));
                *t += 1;
                if passed {
                    *p += 1;
                }
            }

            if !domain_counts.is_empty() {
                let mut rows = BTreeMap::new();
                for (domain, (passed, total)) in &domain_counts {
                    let rate = if *total > 0 {
                        *passed as f64 / *total as f64
                    } else {
                        0.0
                    };
                    rows.insert(
                        domain.clone(),
                        BTreeMap::from([
                            (
                                "pass_rate".to_string(),
                                Score::float(rate, ScoreUnit::Percent)
                                    .display(format!("{:.1}%", rate * 100.0)),
                            ),
                            (
                                "instances".to_string(),
                                Score::integer(*total, ScoreUnit::Count)
                                    .display(format!("{}/{}", passed, total)),
                            ),
                        ]),
                    );
                }
                breakdowns.insert(
                    "By Domain".to_string(),
                    BreakdownTable {
                        title: "Pass Rate by Domain".to_string(),
                        rows,
                    },
                );
            }
        }

        Ok(BenchmarkResult {
            scores,
            breakdowns,
            error_classification: BTreeMap::new(),
            artifacts: vec![],
            diagnostics: vec![],
            raw: raw.clone(),
        })
    }
}

/// Download StableToolBench solvable queries from GitHub
fn download_dataset() -> Result<Vec<SolvableQuery>> {
    let cache_dir = cache_dir()?;
    let data_dir = cache_dir.join("data");

    // Check cache
    if data_dir.exists() {
        return load_instances_from_dir(&data_dir);
    }

    println!("  Downloading StableToolBench solvable queries...");
    fs::create_dir_all(&data_dir)?;

    // Subset files
    let subsets = [
        "G1_instruction",
        "G1_category",
        "G1_tool",
        "G2_category",
        "G2_instruction",
        "G3_instruction",
    ];

    for subset in &subsets {
        let url = format!(
            "https://raw.githubusercontent.com/THUNLP-MT/StableToolBench/main/solvable_queries/{}.json",
            subset
        );
        let path = data_dir.join(format!("{}.json", subset));
        if let Ok(bytes) =
            crate::download::download_with_retry_bytes(&url, 3, 30, "llm-benchmark-runner")
        {
            fs::write(&path, bytes)?;
            println!("    Downloaded {} ({})", subset, path.display());
        } else {
            println!("    Warning: Failed to download {}", subset);
        }
    }

    load_instances_from_dir(&data_dir)
}

fn load_instances_from_dir(data_dir: &PathBuf) -> Result<Vec<SolvableQuery>> {
    let mut all_instances = Vec::new();

    for entry in fs::read_dir(data_dir)? {
        let entry = entry?;
        let path = entry.path();
        if path.extension().map(|e| e == "json").unwrap_or(false) {
            let content = fs::read_to_string(&path)?;
            let subset = path
                .file_stem()
                .map(|s| s.to_string_lossy().to_string())
                .unwrap_or_default();

            // Each file is a JSON object with instances
            let mut instances: Vec<SolvableQuery> = serde_json::from_str(&content)
                .with_context(|| format!("Failed to parse {}", path.display()))?;

            // Tag each instance with its subset name
            for instance in &mut instances {
                instance.subset = Some(subset.clone());
            }

            all_instances.extend(instances);
        }
    }

    Ok(all_instances)
}

fn filter_instances(
    instances: Vec<SolvableQuery>,
    subsets: &Option<Vec<String>>,
    categories: &Option<Vec<String>>,
    num_samples: Option<usize>,
) -> Vec<SolvableQuery> {
    let instances = if let Some(subs) = subsets {
        instances
            .into_iter()
            .filter(|i| {
                if let Some(ref subset) = i.subset {
                    subs.iter()
                        .any(|s| s.to_lowercase() == subset.to_lowercase())
                } else {
                    false
                }
            })
            .collect()
    } else {
        instances
    };

    let instances = if let Some(cats) = categories {
        instances
            .into_iter()
            .filter(|i| {
                if let Some(ref domain) = i.domain {
                    cats.iter()
                        .any(|c| c.to_lowercase() == domain.to_lowercase())
                } else {
                    false
                }
            })
            .collect()
    } else {
        instances
    };

    match num_samples {
        Some(n) => instances.into_iter().take(n).collect(),
        None => instances,
    }
}

fn cache_dir() -> Result<PathBuf> {
    let dir = dirs::cache_dir()
        .ok_or_else(|| anyhow::anyhow!("No cache directory"))?
        .join("llm-benchmark-runner")
        .join("stable_toolbench");
    fs::create_dir_all(&dir)?;
    Ok(dir)
}

/// Per-instance evaluation results
#[derive(Debug)]
#[allow(dead_code)] // fields tracked for future detailed reporting
struct InstanceResult {
    passed: bool,
    tool_selection_correct: bool,
    api_true_positives: usize,
    api_false_positives: usize,
    api_false_negatives: usize,
    param_complete: bool,
    domain: String,
}

/// Evaluate a single instance.
///
/// This uses a **simulation-based** evaluation: we check whether the model selected
/// the correct tools and provided parameters. The StableToolBench paper evaluates with
/// a full pipeline (calling real APIs via MirrorAPI or GPT-based caching, then using
/// GPT-4 as an LLM judge for "Solvable Pass Rate" — SoPR). Our approach measures tool
/// selection accuracy and parameter completeness as a lightweight proxy that does not
/// require API keys or an external evaluator model.
fn evaluate_instance(
    instance: &SolvableQuery,
    idx: usize,
    model: &Model,
    tracker: &mut TokenTracker,
) -> Result<(TaskResult, InstanceResult)> {
    // Build tool definitions for the API
    let tools: Vec<serde_json::Value> = instance
        .api_list
        .iter()
        .map(tool_definition_to_json_schema)
        .collect();

    // System prompt
    let system_prompt =
        "You are a helpful assistant that can call tools/APIs to answer questions. \
        Use the available tools to complete the task. Call only the tools that are relevant.";

    // Call model with tools (single-turn, no history needed)
    let (_text, tool_calls) = tracker.chat_completion_with_tools(
        &model.model_name,
        system_prompt,
        &instance.query,
        tools,
        None,
        false, // use_history: false — single-turn evaluation
    )?;

    // Get ground truth relevant APIs
    let ground_truth: Vec<String> = instance
        .relevant_apis
        .iter()
        .filter_map(|pair| {
            if pair.len() >= 2 {
                Some(format!("{}/{}", pair[0], pair[1]))
            } else if !pair.is_empty() {
                Some(pair[0].clone())
            } else {
                None
            }
        })
        .collect();

    // Evaluate tool calls
    let predicted_apis: Vec<String> = tool_calls.iter().map(|tc| tc.name.clone()).collect();

    // Compute metrics
    let true_positives = predicted_apis
        .iter()
        .filter(|p| {
            ground_truth
                .iter()
                .any(|g| normalize_api(p) == normalize_api(g))
        })
        .count();
    let false_positives = predicted_apis.len() - true_positives;
    let false_negatives = ground_truth.len() - true_positives;

    let tool_selection_correct = true_positives == ground_truth.len() && false_positives == 0;

    // Check parameter completeness (simplified)
    let param_complete = tool_calls.iter().all(|tc| {
        !tc.arguments.is_object()
            || tc
                .arguments
                .as_object()
                .map(|o| !o.is_empty())
                .unwrap_or(false)
    });

    let passed = tool_selection_correct && param_complete;
    let domain = instance
        .domain
        .clone()
        .unwrap_or_else(|| "Unknown".to_string());

    let categories = vec![domain.clone()];

    let task_result = TaskResult::new(
        format!("task-{}", idx),
        passed,
        if passed { 1.0 } else { 0.0 },
        categories,
    )
    .with_metadata(Some(serde_json::json!({
        "tool_selection_correct": tool_selection_correct,
        "api_true_positives": true_positives,
        "api_false_positives": false_positives,
        "api_false_negatives": false_negatives,
        "param_complete": param_complete,
        "domain": domain,
    })));

    let instance_result = InstanceResult {
        passed,
        tool_selection_correct,
        api_true_positives: true_positives,
        api_false_positives: false_positives,
        api_false_negatives: false_negatives,
        param_complete,
        domain,
    };

    Ok((task_result, instance_result))
}

fn tool_definition_to_json_schema(api: &ToolDefinition) -> serde_json::Value {
    let mut properties = serde_json::Map::new();
    let mut required: Vec<String> = Vec::new();

    for param in &api.required_parameters {
        if !param.name.is_empty() {
            properties.insert(
                param.name.clone(),
                serde_json::json!({
                    "type": if param.param_type.is_empty() { "string" } else { &param.param_type },
                    "description": param.description,
                }),
            );
            required.push(param.name.clone());
        }
    }
    for param in &api.optional_parameters {
        if !param.name.is_empty() {
            properties.insert(
                param.name.clone(),
                serde_json::json!({
                    "type": if param.param_type.is_empty() { "string" } else { &param.param_type },
                    "description": param.description,
                }),
            );
        }
    }

    serde_json::json!({
        "type": "function",
        "function": {
            "name": if api.api_name.is_empty() { api.tool_name.clone() } else { api.api_name.clone() },
            "description": api.api_description,
            "parameters": {
                "type": "object",
                "properties": properties,
                "required": required,
            }
        }
    })
}

fn normalize_api(api: &str) -> String {
    api.split('/').next_back().unwrap_or(api).to_lowercase()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_instance(id: u64, domain: Option<&str>, subset: Option<&str>) -> SolvableQuery {
        SolvableQuery {
            api_list: vec![],
            query: format!("q{}", id),
            query_id: id,
            relevant_apis: vec![],
            domain: domain.map(|s| s.into()),
            subset: subset.map(|s| s.into()),
        }
    }

    #[test]
    fn normalize_api_lowercase_last_segment() {
        assert_eq!(normalize_api("toolX/ApiY"), "apiy");
        assert_eq!(normalize_api("standalone"), "standalone");
        assert_eq!(normalize_api("a/b/C/d"), "d");
    }

    #[test]
    fn tool_definition_to_json_schema_minimal() {
        let api = ToolDefinition {
            category_name: "test".into(),
            tool_name: "weather".into(),
            api_name: "get_weather".into(),
            api_description: "Get weather info".into(),
            required_parameters: vec![ParamDef {
                name: "city".into(),
                param_type: "string".into(),
                description: "City name".into(),
                default: "".into(),
            }],
            optional_parameters: vec![],
            method: "GET".into(),
            template_response: None,
        };
        let schema = tool_definition_to_json_schema(&api);
        assert_eq!(schema["function"]["name"], "get_weather");
        assert_eq!(schema["function"]["description"], "Get weather info");
        assert_eq!(
            schema["function"]["parameters"]["required"],
            serde_json::json!(["city"])
        );
    }

    #[test]
    fn tool_definition_uses_tool_name_when_api_name_empty() {
        let api = ToolDefinition {
            category_name: "".into(),
            tool_name: "my_tool".into(),
            api_name: "".into(),
            api_description: "".into(),
            required_parameters: vec![],
            optional_parameters: vec![],
            method: "".into(),
            template_response: None,
        };
        let schema = tool_definition_to_json_schema(&api);
        assert_eq!(schema["function"]["name"], "my_tool");
    }

    #[test]
    fn filter_instances_by_category() {
        let instances = vec![
            make_instance(1, Some("finance"), Some("G1_instruction")),
            make_instance(2, Some("health"), Some("G1_instruction")),
        ];
        let filtered = filter_instances(instances, &None, &Some(vec!["finance".into()]), None);
        assert_eq!(filtered.len(), 1);
        assert_eq!(filtered[0].query_id, 1);
    }

    #[test]
    fn filter_instances_by_subset() {
        let instances = vec![
            make_instance(1, Some("finance"), Some("G1_instruction")),
            make_instance(2, Some("health"), Some("G2_category")),
            make_instance(3, Some("shopping"), Some("G1_instruction")),
        ];
        let filtered =
            filter_instances(instances, &Some(vec!["G1_instruction".into()]), &None, None);
        assert_eq!(filtered.len(), 2);
        assert_eq!(filtered[0].query_id, 1);
        assert_eq!(filtered[1].query_id, 3);
    }

    #[test]
    fn filter_instances_num_samples() {
        let instances = vec![
            make_instance(1, None, Some("G1_instruction")),
            make_instance(2, None, Some("G1_instruction")),
            make_instance(3, None, Some("G1_instruction")),
        ];
        let filtered = filter_instances(instances, &None, &None, Some(2));
        assert_eq!(filtered.len(), 2);
    }

    #[test]
    fn filter_instances_empty_categories_returns_all() {
        let instances = vec![make_instance(1, Some("finance"), Some("G1_instruction"))];
        // None categories means no filtering
        let filtered = filter_instances(instances, &None, &None, None);
        assert_eq!(filtered.len(), 1);
    }

    #[test]
    fn filter_instances_combined_subset_and_category() {
        let instances = vec![
            make_instance(1, Some("finance"), Some("G1_instruction")),
            make_instance(2, Some("health"), Some("G1_instruction")),
            make_instance(3, Some("finance"), Some("G2_category")),
        ];
        let filtered = filter_instances(
            instances,
            &Some(vec!["G1_instruction".into()]),
            &Some(vec!["finance".into()]),
            None,
        );
        assert_eq!(filtered.len(), 1);
        assert_eq!(filtered[0].query_id, 1);
    }
}
