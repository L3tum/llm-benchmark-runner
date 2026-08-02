use crate::benchmarks::Benchmark;
use crate::config::Model;
use crate::shared::{BenchmarkCategory, BenchmarkResult, Score, ScoreUnit, TaskResult};
use crate::token_tracker::TokenTracker;
use anyhow::Result;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use std::collections::BTreeMap;
use std::sync::Mutex;

pub struct RulerBenchmark {
    state: Mutex<RulerState>,
}

struct RulerState {
    items: Vec<RulerItem>,
    current_idx: usize,
}

impl Default for RulerBenchmark {
    fn default() -> Self {
        Self {
            state: Mutex::new(RulerState {
                items: Vec::new(),
                current_idx: 0,
            }),
        }
    }
}

#[derive(Debug, Clone)]
struct RulerItem {
    context: String,
    question: String,
    answer: String,
    context_length: usize, // approximate token count
    task_type: String,     // "niah", "multi_niah", "kv_retrieval", etc.
}

/// Generate synthetic needle-in-haystack tasks.
/// This creates a long context with a specific fact ("needle") buried in irrelevant text ("haystack").
fn generate_ruler_tasks(max_items: usize) -> Vec<RulerItem> {
    let mut rng = StdRng::seed_from_u64(42);
    let mut items = Vec::new();

    // Different context lengths (in approximate words — ~1.3 words per token)
    let context_lengths = [
        1_000,  // ~770 tokens
        2_500,  // ~1900 tokens
        5_000,  // ~3850 tokens
        10_000, // ~7700 tokens
        20_000, // ~15400 tokens
    ];

    // Needle templates
    let needles = [
        ("The secret code is", "8472-XJ-3901"),
        ("The unique identifier is", "UUID-A7F3-92B1-0C4D"),
        ("The answer key is", "42-ALPHA-OMEGA"),
        ("The password is", "S3cur3P@ssw0rd!"),
        ("The hidden number is", "3.14159265358979"),
        ("The verification token is", "VT-9938-2847-1056"),
        ("The magic word is", "HAKUNA MATATA"),
        ("The secret phrase is", "all your base are belong to us"),
        ("The key value is", "42"),
        ("The hidden name is", "Dr. Quantum"),
        ("The target color is", "crimson"),
        ("The special date is", "1999-12-31"),
        ("The coordinate is", "48.8566° N, 2.3522° E"),
    ];

    // Fillers — generic sentences that don't contain any needles
    let fillers = [
        "The weather was pleasant that morning.",
        "She walked down the quiet street.",
        "The book on the shelf was old and dusty.",
        "A cat sat on the windowsill watching birds.",
        "The river flowed gently through the valley.",
        "The old tree provided shade for the garden.",
        "Children played in the park across the road.",
        "The baker opened his shop early each day.",
        "A train passed through the tunnel at noon.",
        "The museum was closed for renovations.",
        "The coffee in the cup was still warm.",
        "The garden had bloomed with summer flowers.",
        "The library was quiet and filled with books.",
        "A rainbow appeared after the afternoon rain.",
        "The mountain stood tall against the sky.",
        "The old clock chimed the hour precisely.",
        "The fish swam peacefully in the pond.",
        "The road wound through the forest.",
        "The stars were visible on the clear night.",
        "The market was bustling with shoppers.",
        "The school bell rang at the end of the day.",
        "The lighthouse guided ships through the fog.",
        "The bakery smelled of fresh bread.",
        "The bicycle leaned against the fence.",
        "The garden gate creaked on its hinges.",
        "The riverbank was lined with willow trees.",
        "The old well had been dry for decades.",
        "The bridge connected the two neighborhoods.",
        "The field was covered in golden wheat.",
        "The chimney released thin streams of smoke.",
        "The pond reflected the morning sky.",
        "The fence marked the boundary of the property.",
        "The lantern glowed softly in the evening.",
        "The pathway led through the meadow.",
        "The window overlooked the garden below.",
        "The staircase spiraled up to the attic.",
        "The bench faced the lake at sunset.",
        "The tower stood at the edge of the cliff.",
        "The courtyard was paved with cobblestones.",
        "The fountain bubbled in the center of the square.",
        "The orchard bore fruit in the autumn.",
        "The harbor was filled with fishing boats.",
        "The castle overlooked the rolling hills.",
        "The meadow was dotted with wildflowers.",
        "The valley echoed with the sound of bells.",
        "The hillside was covered in wild grass.",
        "The cave entrance was hidden by vines.",
        "The lighthouse beam swept across the water.",
        "The meadow stretched to the distant hills.",
        "The forest path was covered in fallen leaves.",
        "The village square was the center of activity.",
        "The old church bell had a deep resonance.",
        "The river delta spread across the coast.",
        "The sand dunes shifted with the wind.",
        "The glacier carved through the mountain valley.",
        "The waterfall cascaded down the rocky cliff.",
        "The hot spring bubbled up from the ground.",
        "The desert stretched as far as the eye could see.",
        "The coral reef teemed with tropical fish.",
        "The canyon walls rose steeply on both sides.",
        "The tundra was flat and treeless.",
        "The mangrove roots tangled in the shallows.",
        "The alpine meadow burst with summer color.",
        "The fjord cut deep into the coastline.",
        "The savanna was home to grazing animals.",
        "The peat bog was dark and waterlogged.",
        "The kelp forest swayed in the ocean current.",
        "The wetland supported diverse bird species.",
        "The dunes shifted slowly over the centuries.",
        "The estuary mixed fresh and salt water.",
    ];

    for context_len in &context_lengths {
        for (prompt, value) in &needles {
            if items.len() >= max_items {
                break;
            }

            // Build context with needle embedded at a random position
            let target_word_count = *context_len;
            let needle = format!("{} {}.", prompt, value);

            // Build the filler list first, then insert needle at random position
            let words_per_filler = 8; // approximate
            let num_fillers = target_word_count / words_per_filler;

            let mut filler_list: Vec<String> = (0..num_fillers)
                .map(|i| fillers[i % fillers.len()].to_string())
                .collect();

            // Insert needle at a random position
            let insert_pos = rng.gen_range(0..filler_list.len().max(1));
            filler_list.insert(insert_pos, needle);

            let context = filler_list.join(" ");
            let question = format!("What {}?", prompt);
            let answer = value.to_string();

            items.push(RulerItem {
                context,
                question,
                answer,
                context_length: target_word_count,
                task_type: "niah".to_string(),
            });
        }
    }

    items
}

impl Benchmark for RulerBenchmark {
    fn name(&self) -> &str {
        "ruler"
    }

    fn display_name(&self) -> &'static str {
        "RULER (Retrieval and Long-context Evaluation)"
    }

    fn category(&self) -> BenchmarkCategory {
        BenchmarkCategory::Research
    }

    fn pre_execute(&self, config: &yaml_serde::Value) -> Result<()> {
        let max_items = config
            .get("max_items")
            .and_then(|v| v.as_u64())
            .unwrap_or(60) as usize;
        let items = generate_ruler_tasks(max_items);
        println!(
            "  RULER: {} synthetic tasks generated (max: {})",
            items.len(),
            max_items
        );
        let mut state = self.state.lock().expect(crate::shared::MUTEX_PANIC_MSG);
        state.items = items;
        state.current_idx = 0;
        Ok(())
    }

    fn execute_one(
        &self,
        model: &Model,
        _config: &yaml_serde::Value,
        tracker: &mut TokenTracker,
    ) -> Result<Option<TaskResult>> {
        let (item, idx) = {
            let mut state = self.state.lock().expect(crate::shared::MUTEX_PANIC_MSG);
            if state.current_idx >= state.items.len() {
                return Ok(None);
            }
            let idx = state.current_idx;
            let item = state.items[idx].clone();
            state.current_idx += 1;
            (item, idx)
        };

        let system_prompt = "You are an information retrieval assistant. Given a long document, answer the question by finding the relevant fact hidden in the text. Output only the answer, nothing else.";

        let user_prompt = format!(
            "Read the following document carefully and answer the question below.\n\n{context}\n\nQuestion: {question}\nAnswer:",
            context = item.context,
            question = item.question
        );

        let response = tracker.chat_completion(&model.model_name, system_prompt, &user_prompt)?;
        let response_trimmed = response.trim();

        // Exact match or substring containment
        let is_correct = response_trimmed.to_lowercase() == item.answer.to_lowercase()
            || response_trimmed
                .to_lowercase()
                .contains(&item.answer.to_lowercase())
            || item
                .answer
                .to_lowercase()
                .contains(&response_trimmed.to_lowercase());

        Ok(Some(
            TaskResult::new(
                format!("task-{}", idx),
                is_correct,
                if is_correct { 1.0 } else { 0.0 },
                vec![item.task_type.clone()],
            )
            .with_metadata(Some(serde_json::json!({
                "context_length": item.context_length,
                "question": item.question,
                "expected": item.answer,
                "response": response_trimmed,
            }))),
        ))
    }

    fn to_report_result(&self, b: &BenchmarkResult) -> Result<BenchmarkResult> {
        let raw = &b.raw;

        let (total, correct, output_tokens, thinking_tokens, length_stats) = {
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

                // Group by context length
                let mut length_stats: BTreeMap<String, (i64, i64)> = BTreeMap::new();
                for task in per_task {
                    if let Some(ctx_len) = task
                        .get("metadata")
                        .and_then(|m| m.get("context_length"))
                        .and_then(|v| v.as_i64())
                    {
                        let key = format!("~{} words", ctx_len);
                        let (p, t) = length_stats.entry(key).or_insert((0, 0));
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
                (total, correct, out, think, length_stats)
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

        // Context length breakdown
        let mut breakdowns = BTreeMap::new();
        if !length_stats.is_empty() {
            let mut rows = BTreeMap::new();
            for (len_key, (len_correct, len_total)) in &length_stats {
                let rate = if *len_total > 0 {
                    *len_correct as f64 / *len_total as f64
                } else {
                    0.0
                };
                rows.insert(
                    len_key.clone(),
                    BTreeMap::from([
                        (
                            "accuracy".to_string(),
                            Score::float(rate * 100.0, ScoreUnit::Percent)
                                .display(format!("{:.1}%", rate * 100.0)),
                        ),
                        (
                            "instances".to_string(),
                            Score::integer(*len_total, ScoreUnit::Count)
                                .display(format!("{}/{}", len_correct, len_total)),
                        ),
                    ]),
                );
            }
            breakdowns.insert(
                "By Context Length".to_string(),
                crate::reports::model::BreakdownTable {
                    title: "Accuracy by Context Length".to_string(),
                    rows,
                },
            );
        }

        Ok(BenchmarkResult {
            scores,
            breakdowns,
            error_classification: BTreeMap::new(),
            artifacts: vec![],
            diagnostics: vec![crate::reports::model::Diagnostic {
                level: "info".to_string(),
                message: format!(
                    "RULER: {}/{} needles found ({:.1}%)",
                    correct,
                    total,
                    accuracy * 100.0
                ),
            }],
            raw: raw.clone(),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_generate_ruler_tasks_deterministic() {
        let tasks1 = generate_ruler_tasks(5);
        let tasks2 = generate_ruler_tasks(5);
        assert_eq!(tasks1.len(), tasks2.len());
        for (t1, t2) in tasks1.iter().zip(tasks2.iter()) {
            assert_eq!(t1.context, t2.context);
        }
    }

    #[test]
    fn test_generate_ruler_tasks_count() {
        // 13 needles × 5 context lengths = 65 max items
        let tasks = generate_ruler_tasks(100);
        assert_eq!(tasks.len(), 65); // capped by needles × context_lengths
    }

    #[test]
    fn test_generate_ruler_tasks_needle_present() {
        let tasks = generate_ruler_tasks(5);
        for task in &tasks {
            assert!(task.context.contains(&task.answer));
        }
    }
}
