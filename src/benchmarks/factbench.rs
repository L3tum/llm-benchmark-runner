use crate::benchmarks::Benchmark;
use crate::config::Model;
use crate::download::download_with_retry_bytes;
use crate::reports::model::{BreakdownTable, Diagnostic};
use crate::shared::{
    fence_prompt_value, BenchmarkCategory, BenchmarkResult, Score, ScoreUnit, TaskResult,
};
use crate::token_tracker::TokenTracker;
use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::fs;
use std::sync::Mutex;

pub struct FactBenchBenchmark {
    state: Mutex<FactBenchState>,
}

struct FactBenchState {
    items: Vec<FactBenchItem>,
    current_idx: usize,
    used_synthetic: bool,
}

impl Default for FactBenchBenchmark {
    fn default() -> Self {
        Self {
            state: Mutex::new(FactBenchState {
                items: Vec::new(),
                current_idx: 0,
                used_synthetic: false,
            }),
        }
    }
}

#[derive(Debug, Clone, Deserialize, Serialize)]
struct FactBenchItem {
    claim: String,
    label: String, // "true", "false", "unknown"
    #[serde(default)]
    source: Option<String>,
    #[serde(default)]
    category: Option<String>,
    #[serde(default)]
    task_id: Option<String>,
}

fn load_factbench_dataset(max_items: usize) -> Vec<FactBenchItem> {
    let cache_dir = dirs::cache_dir()
        .unwrap_or_default()
        .join("llm-benchmark-runner")
        .join("factbench");
    let path = cache_dir.join("factbench.json");

    if path.exists() {
        let content = fs::read_to_string(&path).expect("Failed to read cached FactBench");
        let items: Vec<FactBenchItem> =
            serde_json::from_str(&content).expect("Failed to parse FactBench");
        return items.into_iter().take(max_items).collect();
    }

    fs::create_dir_all(&cache_dir).expect("Failed to create cache dir");
    println!(
        "  Downloading FactBench dataset (up to {} instances)...",
        max_items
    );

    // FactBench — try GitHub release and alternate sources
    // Based on arXiv:2410.22257
    let urls = [
        // Try various potential locations
        "https://raw.githubusercontent.com/factbench/factbench/main/data/test.json",
        "https://raw.githubusercontent.com/factbench/factbench/refs/heads/main/data/test.json",
        "https://huggingface.co/datasets/FactBench/FactBench/resolve/main/test.json",
    ];

    let mut _last_err = None;
    for url in &urls {
        match download_with_retry_bytes(url, 2, 120, "llm-benchmark-runner") {
            Ok(bytes) => {
                // Try direct parse
                if let Ok(items) = serde_json::from_slice::<Vec<FactBenchItem>>(&bytes) {
                    let items: Vec<FactBenchItem> = items.into_iter().take(max_items).collect();
                    fs::write(&path, serde_json::to_string_pretty(&items).unwrap())
                        .expect("Failed to save FactBench");
                    return items;
                }
                // Try nested structure
                if let Ok(val) = serde_json::from_slice::<serde_json::Value>(&bytes) {
                    for key in &["test", "data", "claims", "items", "examples"] {
                        if let Some(arr) = val.get(key).and_then(|v| v.as_array()) {
                            let mut items = Vec::new();
                            for obj in arr.iter().take(max_items) {
                                if let Some(item) = parse_factbench_item(obj) {
                                    items.push(item);
                                }
                            }
                            if !items.is_empty() {
                                fs::write(&path, serde_json::to_string_pretty(&items).unwrap())
                                    .expect("Failed to save FactBench");
                                return items;
                            }
                        }
                    }
                }
                _last_err = Some(anyhow::anyhow!("Failed to parse FactBench from {}", url));
            }
            Err(e) => {
                _last_err = Some(anyhow::anyhow!("Failed to download from {}: {}", url, e));
            }
        }
    }

    // If all downloads fail, generate a synthetic test set for evaluation
    eprintln!(
        "  Warning: Could not download FactBench dataset, using synthetic claims for evaluation."
    );
    let items = generate_synthetic_factbench(max_items);
    fs::write(&path, serde_json::to_string_pretty(&items).unwrap())
        .expect("Failed to save synthetic FactBench");
    items
}

/// Returns true if the synthetic fallback was used.
fn factbench_used_synthetic() -> bool {
    let cache_dir = dirs::cache_dir()
        .unwrap_or_default()
        .join("llm-benchmark-runner")
        .join("factbench");
    let path = cache_dir.join("factbench.json");
    if !path.exists() {
        return false;
    }
    let content = match fs::read_to_string(&path) {
        Ok(c) => c,
        Err(_) => return false,
    };
    let items: Vec<FactBenchItem> = match serde_json::from_str(&content) {
        Ok(i) => i,
        Err(_) => return false,
    };
    items
        .iter()
        .any(|item| item.source.as_deref() == Some("synthetic"))
}

fn parse_factbench_item(obj: &serde_json::Value) -> Option<FactBenchItem> {
    let claim = obj.get("claim")?.as_str()?.to_string();
    let label = obj
        .get("label")
        .and_then(|l| l.as_str())
        .unwrap_or("unknown")
        .to_string();
    let source = obj
        .get("source")
        .and_then(|s| s.as_str())
        .map(|s| s.to_string());
    let category = obj
        .get("category")
        .and_then(|c| c.as_str())
        .map(|c| c.to_string());
    let task_id = obj
        .get("id")
        .and_then(|i| i.as_str())
        .map(|i| i.to_string());

    Some(FactBenchItem {
        claim,
        label,
        source,
        category,
        task_id,
    })
}

/// Generate synthetic fact verification claims when the real dataset is unavailable.
fn generate_synthetic_factbench(max_items: usize) -> Vec<FactBenchItem> {
    let claims = [
        ("The capital of France is Paris.", "true"),
        ("The Earth is flat.", "false"),
        ("Water boils at 100 degrees Celsius at sea level.", "true"),
        ("The Moon orbits the Sun directly.", "false"),
        (
            "The Great Wall of China is visible from space with the naked eye.",
            "false",
        ),
        ("Humans have 23 pairs of chromosomes.", "true"),
        ("Vikings wore horned helmets.", "false"),
        ("The speed of light is approximately 300,000 km/s.", "true"),
        ("Goldfish have a three-second memory.", "false"),
        ("Lightning never strikes the same place twice.", "false"),
        (
            "The Amazon River is the longest river in the world.",
            "false",
        ),
        ("Sharks are mammals.", "false"),
        ("Bats are blind.", "false"),
        ("The human brain uses only 10% of its capacity.", "false"),
        ("Napoleon was short.", "false"),
        ("The Pyramids of Giza were built by aliens.", "false"),
        ("Penicillin was discovered by Alexander Fleming.", "true"),
        ("DNA was discovered by Rosalind Franklin.", "true"),
        ("The first computer bug was a real moth.", "true"),
        ("Einstein failed math in school.", "false"),
        ("The Titanic was unsinkable.", "false"),
        ("Volcanoes exist on Mars.", "true"),
        ("The largest planet in our solar system is Jupiter.", "true"),
        ("Sound can travel through a vacuum.", "false"),
        (
            "The ocean is saltier at the poles than at the equator.",
            "true",
        ),
        ("Octopuses have three hearts.", "true"),
        ("The fastest land animal is the cheetah.", "true"),
        ("Bamboo can grow over 90cm in a single day.", "true"),
        ("The shortest war in history lasted 38 minutes.", "true"),
        ("A group of flamingos is called a flamboyance.", "true"),
        (
            "Mount Everest is the tallest mountain measured from sea level.",
            "true",
        ),
        ("The human eye can see about 10 million colors.", "true"),
        ("Venus is the hottest planet in our solar system.", "true"),
        ("A jellyfish is 95% water.", "true"),
        ("The smallest country in the world is Vatican City.", "true"),
        ("The largest desert in the world is Antarctica.", "true"),
        (
            "There are more stars in the universe than grains of sand on Earth.",
            "true",
        ),
        ("A day on Venus is longer than a year on Venus.", "true"),
        ("The Pacific Ocean is the largest ocean.", "true"),
        ("The longest river in the world is the Nile.", "true"),
        ("Australia is wider than the Moon.", "true"),
        (
            "There are more trees on Earth than stars in the Milky Way.",
            "true",
        ),
        ("The average human has about 7 pounds of bones.", "true"),
        ("A strawberry is not a berry.", "true"),
        ("A banana is a berry.", "true"),
        ("A tomato is a fruit.", "true"),
        ("Glass is a liquid that flows very slowly.", "false"),
        ("The human body has four lungs.", "false"),
        (
            "The heart is located on the right side of the body.",
            "false",
        ),
        ("Mercury is the hottest planet.", "false"),
        ("The Sun is a star.", "true"),
        ("Humans share 50% of their DNA with bananas.", "true"),
        ("The brain is the largest organ in the human body.", "false"),
        ("The skin is the largest organ in the human body.", "true"),
        (
            "The smallest bone in the human body is the stirrup in the ear.",
            "true",
        ),
    ];

    let mut items = Vec::new();
    for (claim, label) in claims.iter().take(max_items) {
        items.push(FactBenchItem {
            claim: claim.to_string(),
            label: label.to_string(),
            source: Some("synthetic".to_string()),
            category: Some("general_knowledge".to_string()),
            task_id: Some(format!("synthetic-{}", items.len())),
        });
    }
    items
}

impl Benchmark for FactBenchBenchmark {
    fn name(&self) -> &str {
        "factbench"
    }

    fn display_name(&self) -> &'static str {
        "FactBench (In-the-Wild Factuality)"
    }

    fn category(&self) -> BenchmarkCategory {
        BenchmarkCategory::Research
    }

    fn pre_execute(&self, config: &yaml_serde::Value) -> Result<()> {
        let max_items = config
            .get("max_items")
            .and_then(|v| v.as_u64())
            .unwrap_or(100) as usize;
        let items = load_factbench_dataset(max_items);
        println!(
            "  FactBench: {} claims loaded (max: {})",
            items.len(),
            max_items
        );
        let mut state = self.state.lock().unwrap_or_else(|p| p.into_inner());
        state.items = items;
        state.current_idx = 0;
        state.used_synthetic = factbench_used_synthetic();
        Ok(())
    }

    fn execute_one(
        &self,
        model: &Model,
        _config: &yaml_serde::Value,
        tracker: &mut TokenTracker,
    ) -> Result<Option<TaskResult>> {
        let (item, idx) = {
            let mut state = self.state.lock().unwrap_or_else(|p| p.into_inner());
            if state.current_idx >= state.items.len() {
                return Ok(None);
            }
            let idx = state.current_idx;
            let item = state.items[idx].clone();
            state.current_idx += 1;
            (item, idx)
        };

        let system_prompt = "You are a fact-checking assistant. Given a claim, determine whether it is TRUE (the claim is factually correct), FALSE (the claim is factually incorrect), or UNKNOWN (the claim cannot be verified or is uncertain). Respond with only TRUE, FALSE, or UNKNOWN.";

        let user_prompt = r#"Claim: The capital of France is Paris.
Answer: TRUE

Claim: The Earth is flat.
Answer: FALSE

Claim: Water boils at 100 degrees Celsius at sea level.
Answer: TRUE

Claim: Lightning never strikes the same place twice.
Answer: FALSE

Claim: The Moon orbits the Sun directly.
Answer: FALSE

Claim: Penicillin was discovered by Alexander Fleming.
Answer: TRUE

Claim: Humans have four lungs.
Answer: FALSE

Claim: The speed of light is approximately 300,000 km/s.
Answer: TRUE

Claim: {claim}
Answer:"#;

        let prompt = user_prompt.replace("{claim}", &fence_prompt_value(&item.claim));
        let response = tracker.chat_completion(&model.model_name, system_prompt, &prompt)?;

        let response_upper = response.trim().to_uppercase();
        let predicted = if response_upper.contains("TRUE") {
            "true"
        } else if response_upper.contains("FALSE") {
            "false"
        } else {
            "unknown"
        };

        let is_correct = predicted == item.label.to_lowercase();

        Ok(Some(
            TaskResult::new(
                format!("task-{}", idx),
                is_correct,
                if is_correct { 1.0 } else { 0.0 },
                vec![item.label.clone()],
            )
            .with_metadata(Some(serde_json::json!({
                "claim": item.claim,
                "expected": item.label,
                "predicted": predicted,
                "response": response.trim(),
            }))),
        ))
    }

    fn to_report_result(&self, b: &BenchmarkResult) -> Result<BenchmarkResult> {
        let raw = &b.raw;

        let (total, correct, output_tokens, thinking_tokens, label_stats) = {
            if let Some(per_task) = raw.get("per_task").and_then(|v| v.as_array()) {
                let total = per_task.len() as i64;
                let correct = per_task
                    .iter()
                    .filter(|t| t.get("passed").and_then(|v| v.as_bool()).unwrap_or(false))
                    .count() as i64;
                let out: i64 = per_task
                    .iter()
                    .filter_map(|t| t.get("output_tokens").and_then(|v| v.as_i64()))
                    .sum();
                let think: i64 = per_task
                    .iter()
                    .filter_map(|t| t.get("thinking_tokens").and_then(|v| v.as_i64()))
                    .sum();

                let mut label_stats: BTreeMap<String, (i64, i64)> = BTreeMap::new();
                for task in per_task {
                    if let Some(label) = task
                        .get("categories")
                        .and_then(|v| v.as_array())
                        .and_then(|arr| arr.first())
                        .and_then(|v| v.as_str())
                    {
                        let (p, t) = label_stats.entry(label.to_string()).or_insert((0, 0));
                        *t += 1;
                        if task
                            .get("passed")
                            .and_then(|v| v.as_bool())
                            .unwrap_or(false)
                        {
                            *p += 1;
                        }
                    }
                }
                (total, correct, out, think, label_stats)
            } else {
                (
                    raw.get("total").and_then(|v| v.as_i64()).unwrap_or(0),
                    raw.get("correct").and_then(|v| v.as_i64()).unwrap_or(0),
                    raw.get("output_tokens")
                        .and_then(|v| v.as_i64())
                        .unwrap_or(0),
                    raw.get("thinking_tokens")
                        .and_then(|v| v.as_i64())
                        .unwrap_or(0),
                    BTreeMap::new(),
                )
            }
        };

        let accuracy = if total > 0 {
            correct as f64 / total as f64
        } else {
            0.0
        };

        let mut scores = BTreeMap::new();
        scores.insert(
            "accuracy".to_string(),
            Score::float(accuracy * 100.0, ScoreUnit::Percent)
                .primary(true)
                .higher_is_better(true),
        );
        scores.insert("total".to_string(), Score::integer(total, ScoreUnit::Count));
        scores.insert(
            "correct".to_string(),
            Score::integer(correct, ScoreUnit::Count),
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

        // Label breakdown
        let mut breakdowns = BTreeMap::new();
        if !label_stats.is_empty() {
            let mut rows = BTreeMap::new();
            for (label, (label_correct, label_total)) in &label_stats {
                let rate = if *label_total > 0 {
                    *label_correct as f64 / *label_total as f64
                } else {
                    0.0
                };
                rows.insert(
                    label.clone(),
                    BTreeMap::from([
                        (
                            "accuracy".to_string(),
                            Score::float(rate * 100.0, ScoreUnit::Percent)
                                .display(format!("{:.1}%", rate * 100.0)),
                        ),
                        (
                            "instances".to_string(),
                            Score::integer(*label_total, ScoreUnit::Count)
                                .display(format!("{}/{}", label_correct, label_total)),
                        ),
                    ]),
                );
            }
            breakdowns.insert(
                "By Label".to_string(),
                BreakdownTable {
                    title: "Accuracy by Label".to_string(),
                    rows,
                },
            );
        }

        Ok(BenchmarkResult {
            scores,
            breakdowns,
            error_classification: BTreeMap::new(),
            artifacts: vec![],
            diagnostics: {
                let mut diags = vec![Diagnostic {
                    level: "info".to_string(),
                    message: format!(
                        "FactBench: {}/{} correct ({:.1}%)",
                        correct,
                        total,
                        accuracy * 100.0
                    ),
                }];
                let state = self.state.lock().unwrap_or_else(|p| p.into_inner());
                if state.used_synthetic {
                    diags.push(Diagnostic {
                        level: "warning".to_string(),
                        message: "FactBench used synthetic fallback data — results are not comparable to official FactBench benchmarks.".to_string(),
                    });
                }
                diags
            },
            raw: raw.clone(),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn test_parse_factbench_item_valid() {
        let obj = json!({
            "id": "test-1",
            "claim": "The sky is blue.",
            "label": "true",
            "source": "wikipedia",
            "category": "science"
        });
        let item = parse_factbench_item(&obj).unwrap();
        assert_eq!(item.claim, "The sky is blue.");
        assert_eq!(item.label, "true");
        assert_eq!(item.source, Some("wikipedia".to_string()));
        assert_eq!(item.category, Some("science".to_string()));
        assert_eq!(item.task_id, Some("test-1".to_string()));
    }

    #[test]
    fn test_parse_factbench_item_missing_fields() {
        let obj = json!({});
        assert!(parse_factbench_item(&obj).is_none());

        let obj = json!({"claim": "test claim"});
        let item = parse_factbench_item(&obj).unwrap();
        assert_eq!(item.label, "unknown");
        assert!(item.source.is_none());
        assert!(item.task_id.is_none());
    }
}
