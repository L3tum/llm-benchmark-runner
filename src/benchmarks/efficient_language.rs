use crate::benchmarks::Benchmark;
use crate::benchmarks::{Difficulty, TranslationState};
use crate::config::Model;
use crate::reports::report_helpers::build_accuracy_report;
use crate::shared::{BenchmarkCategory, BenchmarkResult, TaskResult};
use crate::token_tracker::TokenTracker;
use anyhow::Result;
use rand::prelude::SeedableRng;
use rand::seq::SliceRandom;
use std::sync::{Mutex, OnceLock};

/// Efficient Language Translation benchmark.
/// The model is given a symbol dictionary and encoded sentences, then must decode to English.
/// Three difficulty levels with independent scoring: Easy, Medium, Hard.
pub struct EfficientLanguageBenchmark {
    state: Mutex<TranslationState<EfficientInstance>>,
}

#[derive(Clone)]
struct EfficientInstance {
    difficulty: Difficulty,
    /// Full symbol dictionary: symbol -> English meaning
    full_dictionary: Vec<(String, String)>,
    /// Dictionary shown to the model (may be partial at higher difficulties)
    shown_dictionary: Vec<(String, String)>,
    /// Encoded sentences with expected English translations
    tests: Vec<(String, String)>,
}

impl<'de> serde::Deserialize<'de> for EfficientInstance {
    fn deserialize<D>(deserializer: D) -> std::result::Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        #[derive(serde::Deserialize)]
        struct Raw {
            difficulty: String,
            full_dictionary: Vec<(String, String)>,
            shown_dictionary: Vec<(String, String)>,
            tests: Vec<(String, String)>,
        }
        let raw = Raw::deserialize(deserializer)?;
        let difficulty = match raw.difficulty.as_str() {
            "Easy" => Difficulty::Easy,
            "Medium" => Difficulty::Medium,
            "Hard" => Difficulty::Hard,
            other => panic!("unknown difficulty: {}", other),
        };
        Ok(EfficientInstance {
            difficulty,
            full_dictionary: raw.full_dictionary,
            shown_dictionary: raw.shown_dictionary,
            tests: raw.tests,
        })
    }
}

impl Default for EfficientLanguageBenchmark {
    fn default() -> Self {
        Self {
            state: Mutex::new(TranslationState::default()),
        }
    }
}

/// Load efficient language instances from embedded JSON fixtures.
/// Uses `include_str!` for compile-time embedding — no runtime I/O.
fn generate_instances() -> &'static Vec<EfficientInstance> {
    static INSTANCES: OnceLock<Vec<EfficientInstance>> = OnceLock::new();
    INSTANCES.get_or_init(|| {
        let raw = include_str!("data/efficient_language_instances.json");
        serde_json::from_str(raw)
            .expect("embedded efficient_language_instances.json must be valid JSON")
    })
}

impl Benchmark for EfficientLanguageBenchmark {
    fn name(&self) -> &str {
        "efficient_language"
    }

    fn display_name(&self) -> &'static str {
        "Efficient Language Translation"
    }

    fn category(&self) -> BenchmarkCategory {
        BenchmarkCategory::Reasoning
    }

    fn pre_execute(&self, config: &yaml_serde::Value) -> Result<()> {
        let num_samples = crate::config::extract_usize(config, "num_samples");
        let seed = config.get("seed").and_then(|s| s.as_u64());

        let mut state = self.state.lock().expect(crate::shared::MUTEX_PANIC_MSG);
        let mut instances = generate_instances().clone();

        // Shuffle with optional seed for reproducibility
        let mut rng = if let Some(s) = seed {
            rand::rngs::StdRng::seed_from_u64(s)
        } else {
            rand::rngs::StdRng::from_entropy()
        };
        instances.shuffle(&mut rng);

        if let Some(n) = num_samples {
            if n < instances.len() {
                instances.truncate(n);
            }
        }
        state.instances = instances;
        state.current_idx = 0;
        state.test_idx = 0;
        drop(state);

        Ok(())
    }

    fn execute_one(
        &self,
        model: &Model,
        _config: &yaml_serde::Value,
        tracker: &mut TokenTracker,
    ) -> Result<Option<TaskResult>> {
        // Get the next (instance, test) pair, advancing past multi-test instances
        let Some((instance, test, task_id)) = ({
            let mut state = self.state.lock().expect(crate::shared::MUTEX_PANIC_MSG);
            'outer: loop {
                if state.current_idx >= state.instances.len() {
                    break 'outer None;
                }
                let inst = &state.instances[state.current_idx];
                if state.test_idx < inst.tests.len() {
                    let instance = inst.clone();
                    let test = inst.tests[state.test_idx].clone();
                    let idx = state.current_idx;
                    let tid = state.test_idx;
                    // Advance state after cloning all data we need
                    let tests_len = inst.tests.len();
                    state.test_idx += 1;
                    if state.test_idx >= tests_len {
                        state.current_idx += 1;
                        state.test_idx = 0;
                    }
                    break 'outer Some((instance, test, format!("task-{}-{}", idx, tid)));
                } else {
                    state.current_idx += 1;
                    state.test_idx = 0;
                }
            }
        }) else {
            return Ok(None);
        };

        // Build the prompt
        let mut prompt = String::from(
            "You will be given a shorthand symbol dictionary and an encoded sentence. \
             Decode the encoded sentence to plain English by substituting each symbol with its meaning.\n\n",
        );

        // Add the shown dictionary (may be partial)
        prompt.push_str("### Symbol Dictionary\n");
        for (symbol, meaning) in &instance.shown_dictionary {
            prompt.push_str(&format!("  '{}' → '{}'\n", symbol, meaning));
        }

        // If dictionary is partial, note that
        if instance.shown_dictionary.len() < instance.full_dictionary.len() {
            prompt.push_str(
                "\nNote: The dictionary above is partial. Some symbols may not be defined.\n\
                 Use context and your best understanding to decode the sentence.\n",
            );
        }

        // Add the test
        prompt.push_str("\n### Decode this encoded sentence to English:\n");
        prompt.push_str(&format!("  '{}'", test.0));
        prompt.push_str("\n\nYour decoded English (no quotes):\n");

        let response = tracker.chat_completion(&model.model_name, "", &prompt)?;
        let trimmed = response.trim();

        // Evaluate: check if the response contains the expected translation
        let pass = crate::reports::translation_benchmark::fuzzy_match(trimmed, &test.1);

        let difficulty = instance.difficulty.label();

        Ok(Some(
            TaskResult::new(
                task_id,
                pass,
                if pass { 1.0 } else { 0.0 },
                vec![difficulty.to_string()],
            )
            .with_metadata(Some(serde_json::json!({
                "difficulty": difficulty,
                "test_input": test.0,
                "expected": test.1,
                "response": trimmed,
            }))),
        ))
    }

    fn to_report_result(&self, b: &BenchmarkResult) -> Result<BenchmarkResult> {
        build_accuracy_report(
            b,
            "accuracy",
            "By Difficulty",
            "Accuracy by Difficulty Level",
            None,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn generates_expected_instance_count() {
        let instances = generate_instances();
        assert_eq!(instances.len(), 30);
        assert_eq!(
            instances
                .iter()
                .filter(|i| i.difficulty == Difficulty::Easy)
                .count(),
            10
        );
        assert_eq!(
            instances
                .iter()
                .filter(|i| i.difficulty == Difficulty::Medium)
                .count(),
            10
        );
        assert_eq!(
            instances
                .iter()
                .filter(|i| i.difficulty == Difficulty::Hard)
                .count(),
            10
        );
    }

    #[test]
    fn each_instance_has_tests() {
        let instances = generate_instances();
        for instance in instances {
            assert!(
                !instance.tests.is_empty(),
                "Instance {} has no tests",
                instance.difficulty.label()
            );
        }
    }

    #[test]
    fn easy_instances_have_full_dictionary() {
        let instances = generate_instances();
        for instance in instances {
            if instance.difficulty == Difficulty::Easy {
                assert_eq!(
                    instance.shown_dictionary.len(),
                    instance.full_dictionary.len(),
                    "Easy instance should show full dictionary"
                );
            }
        }
    }

    #[test]
    fn hard_instances_have_sparse_dictionary() {
        let instances = generate_instances();
        for instance in instances {
            if instance.difficulty == Difficulty::Hard {
                assert!(
                    instance.shown_dictionary.len() < instance.full_dictionary.len(),
                    "Hard instance should show fewer symbols than the full dictionary"
                );
            }
        }
    }
}
