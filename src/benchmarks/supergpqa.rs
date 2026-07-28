use crate::benchmarks::Benchmark;
use crate::config::Model;
use crate::download::download_with_retry_bytes;
use crate::reports::model::{BenchmarkCategory, BenchmarkResult, TaskResult};
use crate::token_tracker::TokenTracker;
use anyhow::Result;
use once_cell::sync::Lazy;
use rand::prelude::SliceRandom;
use rand::rngs::StdRng;
use rand::SeedableRng;
use regex::Regex;
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, HashMap};
use std::fs;
use std::path::PathBuf;
use std::sync::Mutex;

/// Single SuperGPQA item from the JSONL dataset.
#[derive(Debug, Clone, Deserialize, Serialize)]
struct SuperGpqaItem {
    uuid: String,
    question: String,
    options: Vec<String>,
    answer: String,
    answer_letter: String,
    discipline: String,
    field: String,
    subfield: String,
    difficulty: String,
    #[serde(default)]
    is_calculation: bool,
}

pub struct SuperGpqaBenchmark {
    state: Mutex<SuperGpqaState>,
}

struct SuperGpqaState {
    items: Vec<SuperGpqaItem>,
    current_idx: usize,
    wrong_classes:
        std::collections::BTreeMap<crate::benchmarks::answer_classifier::WrongAnswerClass, i64>,
}

impl Default for SuperGpqaBenchmark {
    fn default() -> Self {
        Self {
            state: Mutex::new(SuperGpqaState {
                items: Vec::new(),
                current_idx: 0,
                wrong_classes: std::collections::BTreeMap::new(),
            }),
        }
    }
}

fn load_jsonl_data(path: &PathBuf) -> Result<Vec<SuperGpqaItem>> {
    use std::io::{BufRead, BufReader};

    let mut items = Vec::new();
    let file = fs::File::open(path)?;
    let reader = BufReader::new(file);

    for line_result in reader.lines() {
        let line = line_result?;
        if line.is_empty() {
            continue;
        }
        let item: SuperGpqaItem = serde_json::from_str(&line)?;
        items.push(item);
    }
    Ok(items)
}

type GroupedData = (
    HashMap<String, Vec<SuperGpqaItem>>, // discipline
    HashMap<String, Vec<SuperGpqaItem>>, // field
    HashMap<String, Vec<SuperGpqaItem>>, // subfield
    HashMap<String, Vec<SuperGpqaItem>>, // difficulty
);

fn group_all(items: Vec<SuperGpqaItem>) -> GroupedData {
    let mut by_discipline: HashMap<String, Vec<SuperGpqaItem>> = HashMap::new();
    let mut by_field: HashMap<String, Vec<SuperGpqaItem>> = HashMap::new();
    let mut by_subfield: HashMap<String, Vec<SuperGpqaItem>> = HashMap::new();
    let mut by_difficulty: HashMap<String, Vec<SuperGpqaItem>> = HashMap::new();

    for item in items {
        by_discipline
            .entry(item.discipline.clone())
            .or_default()
            .push(item.clone());
        by_field
            .entry(item.field.clone())
            .or_default()
            .push(item.clone());
        by_subfield
            .entry(item.subfield.clone())
            .or_default()
            .push(item.clone());
        by_difficulty
            .entry(item.difficulty.clone())
            .or_default()
            .push(item);
    }

    (by_discipline, by_field, by_subfield, by_difficulty)
}

impl Benchmark for SuperGpqaBenchmark {
    fn name(&self) -> &str {
        "supergpqa"
    }

    fn display_name(&self) -> &'static str {
        "SuperGPQA"
    }

    fn category(&self) -> BenchmarkCategory {
        BenchmarkCategory::Knowledge
    }

    fn pre_execute(&self, config: &yaml_serde::Value) -> Result<()> {
        let num_samples: Option<i64> = config.get("num_samples").and_then(|v| v.as_i64());
        let data_path = self.download_dataset()?;
        let all_items = load_jsonl_data(&data_path)?;

        let subjects_filter = config.get("subjects");
        let subjects: Option<Vec<String>> = match subjects_filter {
            Some(s) if s.is_string() => Some(
                s.as_str()
                    .unwrap()
                    .split(',')
                    .map(|s| s.trim().to_string())
                    .collect(),
            ),
            _ => None,
        };

        let all_data = group_all(all_items);
        let available_disciplines: Vec<String> = all_data.0.keys().cloned().collect();
        let disciplines_to_eval: Vec<String> = if let Some(subj) = &subjects {
            let mut result = Vec::new();
            for s in subj {
                if all_data.0.contains_key(s) {
                    result.push(s.clone());
                } else {
                    eprintln!(
                        "  WARNING: SuperGPQA discipline '{}' not found. Available: {:?}",
                        s, available_disciplines
                    );
                }
            }
            result
        } else {
            available_disciplines
        };

        // Flatten disciplines into single list
        let mut items = Vec::new();
        for disc in &disciplines_to_eval {
            if let Some(questions) = all_data.0.get(disc) {
                let qs = match num_samples {
                    Some(n) if questions.len() > n as usize => questions[..n as usize].to_vec(),
                    _ => questions.clone(),
                };
                items.extend(qs);
            }
        }

        // Shuffle with seed
        let seed = config.get("seed").and_then(|v| v.as_i64()).unwrap_or(42);
        let mut rng = StdRng::seed_from_u64(seed as u64);
        items.shuffle(&mut rng);

        println!("Evaluating SuperGPQA: {} total questions", items.len());

        let mut state = self.state.lock().unwrap();
        state.items = items;
        state.current_idx = 0;
        state.wrong_classes = BTreeMap::new();
        Ok(())
    }

    fn execute_one(
        &self,
        model: &Model,
        _config: &yaml_serde::Value,
        tracker: &mut TokenTracker,
    ) -> Result<Option<TaskResult>> {
        use crate::benchmarks::answer_classifier::classify_wrong_answer;

        let (q, idx) = {
            let mut state = self.state.lock().unwrap();
            if state.current_idx >= state.items.len() {
                return Ok(None);
            }
            let idx = state.current_idx;
            let q = state.items[idx].clone();
            state.current_idx += 1;
            (q, idx)
        };

        let question_text = q.question.clone();
        let choice_map = "ABCD";
        let mut prompt = format!(
            "The following are multiple choice questions (with answers) about {}. Think step by step and then output the answer in the format of \"The answer is (X)\" at the end.\n\n",
            q.discipline
        );
        prompt.push_str(&format!("Question: {}\nOptions: ", question_text));
        for (i, opt) in q.options.iter().enumerate() {
            prompt.push_str(&format!("{}: {}\n", &choice_map[i..i + 1], opt));
        }
        prompt.push_str("Answer: ");

        let response = tracker.chat_completion(&model.model_name, "", &prompt)?;
        let pred = extract_answer(&response);
        let expected = q.answer_letter.chars().next();
        let is_correct = pred == expected;

        if !is_correct {
            let wrong_class =
                classify_wrong_answer(&response, &question_text, expected.unwrap_or('?'), pred);
            let mut state = self.state.lock().unwrap();
            let counter = state.wrong_classes.entry(wrong_class).or_insert(0);
            *counter += 1;
        }

        // Build categories from all dimensions
        let categories = vec![
            q.discipline.clone(),
            q.field.clone(),
            q.subfield.clone(),
            q.difficulty.clone(),
        ];

        Ok(Some(
            TaskResult::new(
                format!("task-{}", idx),
                is_correct,
                if is_correct { 1.0 } else { 0.0 },
                categories,
            )
            .with_metadata(Some(serde_json::json!({
                "question": question_text,
                "expected": expected.map(|c| c.to_string()),
                "predicted": pred.map(|c| c.to_string()),
                "correct": is_correct,
                "discipline": q.discipline,
                "field": q.field,
                "subfield": q.subfield,
                "difficulty": q.difficulty,
            }))),
        ))
    }

    fn to_report_result(&self, b: &BenchmarkResult) -> Result<BenchmarkResult> {
        use crate::benchmarks::answer_classifier::WrongAnswerClass;
        use crate::reports::model::{BreakdownTable, Score, ScoreUnit};

        let raw = &b.raw;

        let (total, correct, _wrong, output_tokens, thinking_tokens) = {
            if let Some(per_task) = raw.get("per_task").and_then(|v| v.as_array()) {
                let total = per_task.len() as i64;
                let correct = per_task
                    .iter()
                    .filter(|t| t.get("passed").and_then(|v| v.as_bool()).unwrap_or(false))
                    .count() as i64;
                let wrong = total - correct;
                let out: i64 = per_task
                    .iter()
                    .filter_map(|t| t.get("output_tokens").and_then(|v| v.as_i64()))
                    .sum();
                let think: i64 = per_task
                    .iter()
                    .filter_map(|t| t.get("thinking_tokens").and_then(|v| v.as_i64()))
                    .sum();
                (total, correct, wrong, out, think)
            } else {
                (
                    raw.get("total_questions")
                        .and_then(|v| v.as_i64())
                        .unwrap_or(0),
                    raw.get("correct").and_then(|v| v.as_i64()).unwrap_or(0),
                    raw.get("wrong").and_then(|v| v.as_i64()).unwrap_or(0),
                    raw.get("output_tokens")
                        .and_then(|v| v.as_i64())
                        .unwrap_or(0),
                    raw.get("thinking_tokens")
                        .and_then(|v| v.as_i64())
                        .unwrap_or(0),
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
        scores.insert(
            "total_questions".to_string(),
            Score::integer(total, ScoreUnit::Count),
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

        // Build breakdowns from per_task categories
        let mut breakdowns = BTreeMap::new();
        if let Some(per_task) = raw.get("per_task").and_then(|v| v.as_array()) {
            for dim in &["discipline", "field", "subfield", "difficulty"] {
                let mut cat_map: BTreeMap<String, (i64, i64)> = BTreeMap::new();
                for task in per_task {
                    if let Some(meta) = task.get("metadata").and_then(|v| v.as_object()) {
                        if let Some(cat) = meta.get(*dim).and_then(|v| v.as_str()) {
                            let entry = cat_map.entry(cat.to_string()).or_insert((0, 0));
                            entry.1 += 1;
                            if task
                                .get("passed")
                                .and_then(|v| v.as_bool())
                                .unwrap_or(false)
                            {
                                entry.0 += 1;
                            }
                        }
                    }
                }
                let mut rows = BTreeMap::new();
                for (cat, (corr, t)) in cat_map {
                    let acc = if t > 0 { corr as f64 / t as f64 } else { 0.0 };
                    rows.insert(
                        cat,
                        BTreeMap::from([
                            (
                                "accuracy".to_string(),
                                Score::float(acc * 100.0, ScoreUnit::Percent),
                            ),
                            (
                                "correct".to_string(),
                                Score::integer(corr, ScoreUnit::Count),
                            ),
                            (
                                "wrong".to_string(),
                                Score::integer(t - corr, ScoreUnit::Count),
                            ),
                        ]),
                    );
                }
                if !rows.is_empty() {
                    breakdowns.insert(
                        dim.to_string(),
                        BreakdownTable {
                            title: format!("{} Breakdown", dim.to_ascii_uppercase()),
                            rows,
                        },
                    );
                }
            }
        }

        // Error classification
        let error_classification: BTreeMap<WrongAnswerClass, i64> = raw
            .get("error_classification")
            .and_then(|v| v.as_object())
            .map(|obj| {
                obj.iter()
                    .filter_map(|(key, val)| {
                        let class = match key.as_str() {
                            "Wrong Answer Key" => Some(WrongAnswerClass::WrongAnswerKey),
                            "Invalid Answer Key" => Some(WrongAnswerClass::InvalidAnswerKey),
                            "No Answer" => Some(WrongAnswerClass::NoAnswer),
                            "Uncertainty" => Some(WrongAnswerClass::Uncertainty),
                            "Refused" => Some(WrongAnswerClass::Refused),
                            "Looping" => Some(WrongAnswerClass::Looping),
                            "Truncated" => Some(WrongAnswerClass::Truncated),
                            "Off-Topic / Hallucination" => Some(WrongAnswerClass::OffTopic),
                            _ => None,
                        };
                        class.zip(val.as_i64())
                    })
                    .collect()
            })
            .unwrap_or_default();

        Ok(BenchmarkResult {
            scores,
            breakdowns,
            error_classification,
            artifacts: vec![],
            diagnostics: vec![],
            raw: raw.clone(),
        })
    }
}

impl SuperGpqaBenchmark {
    pub fn download_dataset(&self) -> Result<PathBuf> {
        let cache_dir = dirs::cache_dir()
            .unwrap_or_default()
            .join("llm-benchmark-runner")
            .join("supergpqa");
        fs::create_dir_all(&cache_dir)?;
        let path = cache_dir.join("SuperGPQA-all.jsonl");
        if path.exists() {
            return Ok(path);
        }

        let url =
            "https://huggingface.co/datasets/m-a-p/SuperGPQA/resolve/main/SuperGPQA-all.jsonl";
        println!("  Downloading SuperGPQA dataset (with retry + timeout)...");

        // Use shared download utility with retry and timeout
        let bytes = download_with_retry_bytes(url, 3, 120, "llm-benchmark-runner")?;
        fs::write(&path, bytes)?;

        Ok(path)
    }
}

// Precompiled regexes to avoid repeated compilation overhead during evaluation
static RE_ANSWER_IS: Lazy<Regex> =
    Lazy::new(|| Regex::new(r"\banswer is\s*\(?([A-J])\)?").unwrap());
static RE_ANSWER_COLON: Lazy<Regex> = Lazy::new(|| Regex::new(r"[aA]nswer:\s*([A-J])").unwrap());
static RE_LETTER: Lazy<Regex> = Lazy::new(|| Regex::new(r"\b([A-J])\b").unwrap());
static RE_SEQUENCE: Lazy<Regex> =
    Lazy::new(|| Regex::new(r"\b[A-J]\b\s*[,;]\s*\b[A-J]\b").unwrap());

fn extract_answer(text: &str) -> Option<char> {
    // Scan entire text for answer patterns, use the last match
    let last = RE_ANSWER_IS.captures_iter(text).last();
    if let Some(caps) = last {
        if let Some(m) = caps.get(1) {
            return m.as_str().chars().next();
        }
    }

    let last = RE_ANSWER_COLON.captures_iter(text).last();
    if let Some(caps) = last {
        if let Some(m) = caps.get(1) {
            return m.as_str().chars().next();
        }
    }

    // Final fallback: find the last single letter from A-J that isn't part of a sequence.
    // For pronouns ("I", "A", "J"), only accept if the letter is the last word in the text,
    // to avoid false positives from sentences like "I think the answer is C".
    let sequences = RE_SEQUENCE
        .find_iter(text)
        .map(|m| (m.start(), m.end()))
        .collect::<Vec<_>>();

    let mut last_letter = None;
    for caps in RE_LETTER.captures_iter(text) {
        if let Some(letter_match) = caps.get(1) {
            let start = letter_match.start();
            let letter = letter_match.as_str().chars().next().unwrap_or(' ');
            let in_sequence = sequences.iter().any(|(s, e)| start >= *s && start < *e);

            if !in_sequence {
                // For pronouns (I, A, J), only accept if it's the last word in the text.
                // A word is last if the text after the match contains no more alphabetic characters
                // (i.e., only whitespace and punctuation remain).
                let is_pronoun = letter == 'I' || letter == 'A' || letter == 'J';
                let is_last_word = !text[start + 1..].chars().any(|c| c.is_alphabetic());

                if !is_pronoun || is_last_word {
                    last_letter = Some(letter);
                }
            }
        }
    }
    last_letter
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    use tempfile::tempdir;

    // Helper to create a JSONL file with items for testing
    type JsonlItem = (
        &'static str,
        &'static str,
        Vec<&'static str>,
        &'static str,
        &'static str,
        &'static str,
        &'static str,
    );
    fn create_test_jsonl(dir: &std::path::Path, items: &[JsonlItem]) -> PathBuf {
        let path = dir.join("test.jsonl");
        let mut file = fs::File::create(&path).unwrap();
        for (uuid, question, options, answer_letter, discipline, field, subfield) in items {
            let item = SuperGpqaItem {
                uuid: uuid.to_string(),
                question: question.to_string(),
                options: options.iter().map(|s| s.to_string()).collect(),
                answer: "A".to_string(),
                answer_letter: answer_letter.to_string(),
                discipline: discipline.to_string(),
                field: field.to_string(),
                subfield: subfield.to_string(),
                difficulty: "middle".to_string(),
                is_calculation: false,
            };
            writeln!(file, "{}", serde_json::to_string(&item).unwrap()).unwrap();
        }
        path
    }

    #[test]
    fn test_extract_answer_pattern_answer_is_paren() {
        assert_eq!(extract_answer("The answer is (C)."), Some('C'));
        assert_eq!(
            extract_answer("The answer is (D). So that's correct."),
            Some('D')
        );
    }

    #[test]
    fn test_extract_answer_pattern_answer_is_no_paren() {
        assert_eq!(extract_answer("The answer is A."), Some('A'));
        assert_eq!(extract_answer("The answer is B, I think."), Some('B'));
    }

    #[test]
    fn test_extract_answer_pattern_answer_colon() {
        assert_eq!(extract_answer("Answer: C"), Some('C'));
        assert_eq!(extract_answer("answer: D"), Some('D'));
    }

    #[test]
    fn test_extract_answer_pattern_mixed() {
        // "answer is" takes priority
        assert_eq!(extract_answer("Answer: A\nThe answer is (B)"), Some('B'));
    }

    #[test]
    fn test_extract_answer_fallback_single_letter() {
        assert_eq!(
            extract_answer("Therefore, B is the right choice."),
            Some('B')
        );
        assert_eq!(
            extract_answer("Options: A) foo B) bar\n\nB is correct."),
            Some('B')
        );
    }

    #[test]
    fn test_extract_answer_ignores_sequence() {
        // A sequence "A, B" should be skipped
        assert_eq!(
            extract_answer("Options are A, B. The answer is C."),
            Some('C')
        );
        // Only the sequence, no other match — all letters are in sequences, so None
        assert_eq!(extract_answer("Options A, B, C, D."), None);
    }

    #[test]
    fn test_extract_answer_no_match() {
        // "I" as a pronoun in the middle of a sentence should not be picked up
        assert_eq!(extract_answer("I don't know."), None);
        assert_eq!(extract_answer("The sky is blue."), None);
        // "A" as a pronoun in the middle should not be picked up
        assert_eq!(extract_answer("A is the correct answer."), None);
    }

    #[test]
    fn test_extract_answer_last_word_pronoun() {
        // If a pronoun is the last word (trailing punctuation only), it can be a valid answer
        assert_eq!(extract_answer("The answer is A."), Some('A'));
        assert_eq!(extract_answer("Maybe J."), Some('J'));
        // "I" as the last word (sentence ends with "I.") should be accepted
        assert_eq!(extract_answer("I think I know the answer, I."), Some('I'));
    }

    #[test]
    fn test_extract_answer_pronoun_not_last() {
        // "I" as the first word (not last) should not be accepted
        assert_eq!(extract_answer("I think."), None);
    }

    #[test]
    fn test_extract_answer_last_match_wins() {
        assert_eq!(
            extract_answer("The answer is (A) but wait, the answer is (B)."),
            Some('B')
        );
    }

    #[test]
    fn test_load_jsonl_data() {
        let dir = tempdir().unwrap();
        let path = create_test_jsonl(
            dir.path(),
            &[
                (
                    "1",
                    "What is 2+2?",
                    vec!["3", "4", "5"],
                    "B",
                    "Mathematics",
                    "Algebra",
                    "Elementary Algebra",
                ),
                (
                    "2",
                    "What is 3+3?",
                    vec!["5", "6", "7"],
                    "B",
                    "Mathematics",
                    "Algebra",
                    "Elementary Algebra",
                ),
            ],
        );
        let items = load_jsonl_data(&path).unwrap();
        assert_eq!(items.len(), 2);
        assert_eq!(items[0].uuid, "1");
        assert_eq!(items[1].answer_letter, "B");
    }

    #[test]
    fn test_load_jsonl_empty_lines_ignored() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("test.jsonl");
        let mut file = fs::File::create(&path).unwrap();
        writeln!(file).unwrap();
        let item = SuperGpqaItem {
            uuid: "1".to_string(),
            question: "Q".to_string(),
            options: vec!["A".to_string()],
            answer: "A".to_string(),
            answer_letter: "A".to_string(),
            discipline: "D".to_string(),
            field: "F".to_string(),
            subfield: "S".to_string(),
            difficulty: "e".to_string(),
            is_calculation: false,
        };
        writeln!(file, "{}", serde_json::to_string(&item).unwrap()).unwrap();
        writeln!(file).unwrap();
        let items = load_jsonl_data(&path).unwrap();
        assert_eq!(items.len(), 1);
    }

    #[test]
    fn test_group_all_single_pass() {
        let items = vec![
            SuperGpqaItem {
                uuid: "1".to_string(),
                question: "Q".to_string(),
                options: vec![],
                answer: "A".to_string(),
                answer_letter: "A".to_string(),
                discipline: "Math".to_string(),
                field: "Algebra".to_string(),
                subfield: "Elem".to_string(),
                difficulty: "easy".to_string(),
                is_calculation: false,
            },
            SuperGpqaItem {
                uuid: "2".to_string(),
                question: "Q2".to_string(),
                options: vec![],
                answer: "A".to_string(),
                answer_letter: "A".to_string(),
                discipline: "Biology".to_string(),
                field: "Bio".to_string(),
                subfield: "Mol".to_string(),
                difficulty: "hard".to_string(),
                is_calculation: false,
            },
            SuperGpqaItem {
                uuid: "3".to_string(),
                question: "Q3".to_string(),
                options: vec![],
                answer: "A".to_string(),
                answer_letter: "A".to_string(),
                discipline: "Math".to_string(),
                field: "Geometry".to_string(),
                subfield: "Elem".to_string(),
                difficulty: "easy".to_string(),
                is_calculation: false,
            },
        ];
        let (by_discipline, by_field, by_subfield, by_difficulty) = group_all(items);

        // Discipline
        assert_eq!(by_discipline.len(), 2);
        assert_eq!(by_discipline["Math"].len(), 2);
        assert_eq!(by_discipline["Biology"].len(), 1);

        // Field
        assert_eq!(by_field["Algebra"].len(), 1);
        assert_eq!(by_field["Geometry"].len(), 1);
        assert_eq!(by_field["Bio"].len(), 1);

        // Subfield
        assert_eq!(by_subfield["Elem"].len(), 2);
        assert_eq!(by_subfield["Mol"].len(), 1);

        // Difficulty
        assert_eq!(by_difficulty["easy"].len(), 2);
        assert_eq!(by_difficulty["hard"].len(), 1);
    }
}
