use crate::benchmarks::Benchmark;
use crate::benchmarks::{Difficulty, TranslationState};
use crate::config::Model;
use crate::reports::report_helpers::build_accuracy_report;
use crate::reports::translation_benchmark::fuzzy_match;
use crate::shared::{BenchmarkCategory, BenchmarkResult, TaskResult};
use crate::token_tracker::TokenTracker;
use anyhow::Result;
use rand::prelude::SeedableRng;
use rand::seq::SliceRandom;
use std::sync::{Mutex, OnceLock};

/// The model is given a primer (examples) and must translate fictional sentences to English.
/// Three difficulty levels with independent scoring: Easy, Medium, Hard.
pub struct FictionalLanguageBenchmark {
    state: Mutex<TranslationState<FictionalInstance>>,
}

#[derive(Clone)]
struct FictionalInstance {
    difficulty: Difficulty,
    /// Vocabulary: fictional_word -> English word
    vocabulary: Vec<(String, String)>,
    /// Grammar rules shown in the prompt (may be partial at higher difficulties)
    grammar_rules: String,
    /// Primer examples: (fictional sentence, English translation)
    primer: Vec<(String, String)>,
    /// Test questions: (fictional sentence, expected English translation)
    tests: Vec<(String, String)>,
}

impl<'de> serde::Deserialize<'de> for FictionalInstance {
    fn deserialize<D>(deserializer: D) -> std::result::Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        #[derive(serde::Deserialize)]
        struct Raw {
            difficulty: String,
            vocabulary: Vec<(String, String)>,
            grammar_rules: String,
            primer: Vec<(String, String)>,
            tests: Vec<(String, String)>,
        }
        let raw = Raw::deserialize(deserializer)?;
        let difficulty = match raw.difficulty.as_str() {
            "Easy" => Difficulty::Easy,
            "Medium" => Difficulty::Medium,
            "Hard" => Difficulty::Hard,
            other => {
                return Err(serde::de::Error::unknown_variant(
                    other,
                    &["Easy", "Medium", "Hard"],
                ));
            }
        };
        Ok(FictionalInstance {
            difficulty,
            vocabulary: raw.vocabulary,
            grammar_rules: raw.grammar_rules,
            primer: raw.primer,
            tests: raw.tests,
        })
    }
}

impl Default for FictionalLanguageBenchmark {
    fn default() -> Self {
        Self {
            state: Mutex::new(TranslationState::default()),
        }
    }
}

/// Load fictional language instances from embedded JSON fixtures.
/// Uses `include_str!` for compile-time embedding — no runtime I/O.
fn generate_instances() -> &'static Vec<FictionalInstance> {
    static INSTANCES: OnceLock<Vec<FictionalInstance>> = OnceLock::new();
    INSTANCES.get_or_init(|| {
        let raw = include_str!("data/fictional_language_instances.json");
        serde_json::from_str(raw)
            .expect("embedded fictional_language_instances.json must be valid JSON")
    })
}

impl Benchmark for FictionalLanguageBenchmark {
    fn name(&self) -> &str {
        "fictional_language"
    }

    fn display_name(&self) -> &'static str {
        "Fictional Language Translation"
    }

    fn category(&self) -> BenchmarkCategory {
        BenchmarkCategory::Reasoning
    }

    fn pre_execute(&self, config: &yaml_serde::Value) -> Result<()> {
        let num_samples = crate::config::extract_usize(config, "num_samples");
        let seed = config.get("seed").and_then(|s| s.as_u64());

        let mut rng = seed.map(rand::rngs::StdRng::seed_from_u64);

        let mut instances = generate_instances().clone();

        if let Some(n) = num_samples {
            instances.truncate(n);
        }

        if let Some(ref mut r) = rng {
            instances.shuffle(r);
        }

        println!(
            "Fictional Language: {} instances (seed={})",
            instances.len(),
            seed.map_or("none".to_string(), |s| s.to_string())
        );

        let mut state = self.state.lock().unwrap_or_else(|p| p.into_inner());
        state.instances = instances;
        state.current_idx = 0;
        drop(state);

        Ok(())
    }

    fn execute_one(
        &self,
        model: &Model,
        _config: &yaml_serde::Value,
        tracker: &mut TokenTracker,
    ) -> Result<Option<TaskResult>> {
        let (instance, idx) = {
            let mut state = self.state.lock().unwrap_or_else(|p| p.into_inner());
            if state.current_idx >= state.instances.len() {
                return Ok(None);
            }
            let idx = state.current_idx;
            let instance = state.instances[idx].clone();
            state.current_idx += 1;
            (instance, idx)
        };

        let system_prompt = "You are a translation expert. Given examples of a fictional language, translate new sentences to English.";

        // Build prompt from vocabulary, grammar rules, and primer
        let mut prompt = String::new();

        prompt.push_str("Here is a fictional language with the following vocabulary:\n");
        for (word, translation) in &instance.vocabulary {
            prompt.push_str(&format!("  {} -> {}\n", word, translation));
        }

        prompt.push_str("\nGrammar rules:\n");
        prompt.push_str(&instance.grammar_rules);

        prompt.push_str("\nExamples:\n");
        for (fictional, english) in &instance.primer {
            prompt.push_str(&format!("  '{}' -> '{}'\n", fictional, english));
        }

        prompt.push_str("\nNow translate the following sentences:\n");
        let test_prompts: Vec<&str> = instance.tests.iter().map(|t| t.0.as_str()).collect();
        prompt.push_str(&format!("  {}\n", test_prompts.join("\n  ")));
        prompt.push_str("Provide your translations in order, one per line.");

        let response = tracker.chat_completion(&model.model_name, system_prompt, &prompt)?;

        // Evaluate responses against expected translations
        let responses: Vec<&str> = response.lines().collect();
        let total = instance.tests.len();
        let mut correct = 0;

        for (i, (_test, expected)) in instance.tests.iter().enumerate() {
            if let Some(response) = responses.get(i) {
                if fuzzy_match(response.trim(), expected) {
                    correct += 1;
                }
            }
        }

        let score = if total > 0 {
            correct as f64 / total as f64
        } else {
            0.0
        };

        Ok(Some(
            TaskResult::new(
                format!("test-{}", idx),
                score == 1.0,
                score,
                vec![instance.difficulty.label().to_string()],
            )
            .with_metadata(Some(serde_json::json!({
                "difficulty": instance.difficulty.label(),
                "correct": correct,
                "total": total,
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
    fn easy_6_vocabulary_consistent() {
        let instances = generate_instances();
        // Easy 6: Colors and objects — vocabulary must match primer expectations
        let easy_6 = instances
            .iter()
            .enumerate()
            .find(|(_, i)| {
                i.difficulty == Difficulty::Easy && i.tests.iter().any(|t| t.1 == "green car")
            })
            .expect("Easy 6 (colors) instance should exist");
        // "blu" should map to "blue" (not "red")
        assert!(
            easy_6
                .1
                .vocabulary
                .iter()
                .any(|(k, v)| *k == "blu" && *v == "blue"),
            "blu should map to blue"
        );
        // "red" should map to "red" (not "blue")
        assert!(
            easy_6
                .1
                .vocabulary
                .iter()
                .any(|(k, v)| *k == "red" && *v == "red"),
            "red should map to red"
        );
    }

    #[test]
    fn easy_3_arithmetic_correct() {
        let instances = generate_instances();
        // Easy 3: Number words — arithmetic must be correct
        let easy_3 = instances
            .iter()
            .find(|i| {
                i.difficulty == Difficulty::Easy && i.vocabulary.iter().any(|(k, _)| *k == "uno")
            })
            .expect("Easy 3 (numbers) instance should exist");
        // "two plus two equals four" is correct; "two plus two equals three" is wrong
        assert!(
            easy_3
                .primer
                .iter()
                .all(|(_, eng)| eng != "two plus two equals three"),
            "No false arithmetic in primer"
        );
        assert!(
            easy_3
                .primer
                .iter()
                .any(|(_, eng)| eng == "two plus two equals four"),
            "two plus two equals four should be in primer"
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
            assert!(
                !instance.primer.is_empty(),
                "Instance {} has no primer examples",
                instance.difficulty.label()
            );
        }
    }

    #[test]
    fn medium_instances_have_partial_vocabulary() {
        let medium: Vec<_> = generate_instances()
            .iter()
            .filter(|i| i.difficulty == Difficulty::Medium)
            .cloned()
            .collect();
        for inst in &medium {
            assert!(
                !inst.vocabulary.is_empty(),
                "Medium instance should have some vocabulary"
            );
        }
    }

    #[test]
    fn hard_instances_have_sparse_grammar() {
        let hard: Vec<_> = generate_instances()
            .iter()
            .filter(|i| i.difficulty == Difficulty::Hard)
            .cloned()
            .collect();
        for inst in &hard {
            assert!(
                !inst.grammar_rules.is_empty(),
                "Hard instance should have grammar rules (may be sparse)"
            );
        }
    }
}
