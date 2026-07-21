use crate::client::Client;
use crate::config::Model;
use crate::download::download_with_retry_bytes;
use crate::reports::model::{BenchmarkCategory, BenchmarkResult, BreakdownTable, Score, ScoreUnit};
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

pub struct SuperGpqaBenchmark;

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

impl super::Benchmark for SuperGpqaBenchmark {
    fn name(&self) -> &str {
        "supergpqa"
    }

    fn display_name(&self) -> &'static str {
        "SuperGPQA"
    }

    fn category(&self) -> BenchmarkCategory {
        BenchmarkCategory::Knowledge
    }

    fn pre_execute(&self, _config: &yaml_serde::Value) -> Result<()> {
        self.download_dataset()?;
        Ok(())
    }

    fn to_report_result(&self, b: &BenchmarkResult) -> Result<BenchmarkResult> {
        let raw = &b.raw;
        let accuracy = raw.get("accuracy").and_then(|v| v.as_f64()).unwrap_or(0.0);
        let total_questions = raw
            .get("total_questions")
            .and_then(|v| v.as_i64())
            .unwrap_or(0);
        let output_tokens = raw
            .get("output_tokens")
            .and_then(|v| v.as_i64())
            .unwrap_or(0);
        let thinking_tokens = raw
            .get("thinking_tokens")
            .and_then(|v| v.as_i64())
            .unwrap_or(0);

        let mut scores = BTreeMap::new();
        scores.insert(
            "accuracy".to_string(),
            Score::float(accuracy, ScoreUnit::Percent)
                .primary(true)
                .higher_is_better(true),
        );
        scores.insert(
            "total_questions".to_string(),
            Score::integer(total_questions, ScoreUnit::Count),
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

        // Build breakdown tables
        let mut breakdowns = BTreeMap::new();

        // Helper to create a breakdown table from a results JSON object
        fn build_breakdown_table(title: &str, data: &serde_json::Value) -> BreakdownTable {
            let mut rows = BTreeMap::new();
            if let Some(obj) = data.as_object() {
                for (key, val) in obj {
                    if let Some(obj) = val.as_object() {
                        let acc = obj.get("acc").and_then(|v| v.as_f64()).unwrap_or(0.0);
                        let correct = obj.get("corr").and_then(|v| v.as_i64()).unwrap_or(0);
                        let wrong = obj.get("wrong").and_then(|v| v.as_i64()).unwrap_or(0);
                        let mut row_scores = BTreeMap::new();
                        row_scores.insert(
                            "accuracy".to_string(),
                            Score::float(acc, ScoreUnit::Percent),
                        );
                        row_scores.insert(
                            "correct".to_string(),
                            Score::integer(correct, ScoreUnit::Count),
                        );
                        row_scores
                            .insert("wrong".to_string(), Score::integer(wrong, ScoreUnit::Count));
                        rows.insert(key.clone(), row_scores);
                    }
                }
            }
            BreakdownTable {
                title: title.to_string(),
                rows,
            }
        }

        // Add breakdowns for discipline (primary), field, subfield, and difficulty
        if let Some(discipline_data) = raw.get("results_by_discipline") {
            breakdowns.insert(
                "discipline".to_string(),
                build_breakdown_table("Discipline Breakdown", discipline_data),
            );
        }
        if let Some(field_data) = raw.get("results_by_field") {
            breakdowns.insert(
                "field".to_string(),
                build_breakdown_table("Field Breakdown", field_data),
            );
        }
        if let Some(subfield_data) = raw.get("results_by_subfield") {
            breakdowns.insert(
                "subfield".to_string(),
                build_breakdown_table("Subfield Breakdown", subfield_data),
            );
        }
        if let Some(difficulty_data) = raw.get("results_by_difficulty") {
            breakdowns.insert(
                "difficulty".to_string(),
                build_breakdown_table("Difficulty Breakdown", difficulty_data),
            );
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

    fn execute(&self, model: &Model, config: &yaml_serde::Value) -> Result<BenchmarkResult> {
        let client = Client::new_with_model_params(&model.proxy, model.set_params.as_ref())?;
        let num_samples: Option<i64> = config.get("num_samples").and_then(|v| v.as_i64());
        let subjects_filter = config.get("subjects");
        let subjects: Option<Vec<String>> = match subjects_filter {
            Some(s) if s.is_string() => Some(
                s.as_str()
                    .unwrap()
                    .split(',')
                    .map(|s| s.trim().to_string())
                    .collect(),
            ),
            Some(s) if s.is_null() => None,
            _ => None,
        };

        let data_path = self.download_dataset()?;
        let all_items = load_jsonl_data(&data_path)?;

        // Group data by discipline, field, subfield, and difficulty in a single pass
        let (by_discipline, by_field, by_subfield, _by_difficulty) = group_all(all_items);

        // Determine which disciplines/fields/subfields to evaluate
        let subjects_to_eval: Vec<String> = if let Some(subj) = subjects {
            // Collect matching disciplines, fields, and subfields
            let mut result = Vec::new();
            let all_disciplines: Vec<String> = by_discipline.keys().cloned().collect();
            let all_fields: Vec<String> = by_field.keys().cloned().collect();
            let all_subfields: Vec<String> = by_subfield.keys().cloned().collect();

            for s in &subj {
                if by_discipline.contains_key(s)
                    || by_field.contains_key(s)
                    || by_subfield.contains_key(s)
                {
                    result.push(s.clone());
                } else {
                    eprintln!(
                        "  WARNING: SuperGPQA category '{}' not found, skipping. Available disciplines: {:?}, fields: {:?}, subfields: {:?}",
                        s, all_disciplines, all_fields, all_subfields
                    );
                }
            }
            result
        } else {
            // Use all available disciplines by default
            by_discipline.keys().cloned().collect()
        };

        // Evaluate all matching questions (by discipline or field or subfield)
        let mut total_questions = 0usize;
        let mut total_output_tokens: u64 = 0;
        let mut total_thinking_tokens: u64 = 0;

        // Group results by discipline, field, subfield, and difficulty for reporting

        // Collect all questions to evaluate based on subject filter
        let mut all_questions: Vec<SuperGpqaItem> = Vec::new();
        for subject in &subjects_to_eval {
            // Check if it matches a discipline
            if let Some(questions) = by_discipline.get(subject) {
                for q in questions {
                    if !all_questions.iter().any(|item| item.uuid == q.uuid) {
                        all_questions.push(q.clone());
                    }
                }
            } else if let Some(questions) = by_field.get(subject) {
                for q in questions {
                    if !all_questions.iter().any(|item| item.uuid == q.uuid) {
                        all_questions.push(q.clone());
                    }
                }
            } else if let Some(questions) = by_subfield.get(subject) {
                for q in questions {
                    if !all_questions.iter().any(|item| item.uuid == q.uuid) {
                        all_questions.push(q.clone());
                    }
                }
            }
        }

        // Read optional seed for reproducible shuffling (seed=0 is default for determinism)
        let seed: u64 = config
            .get("seed")
            .and_then(|v| v.as_i64())
            .map(|s| s as u64)
            .unwrap_or(0);

        // Apply num_samples if set
        let questions: Vec<SuperGpqaItem> = match num_samples {
            Some(n) if all_questions.len() > n as usize => {
                let mut questions_vec = all_questions.clone();
                // Use seed-based shuffle for reproducible sampling (use UUID as fallback)
                let mut rng = StdRng::seed_from_u64(seed);
                questions_vec.shuffle(&mut rng);
                questions_vec[..n as usize].to_vec()
            }
            _ => all_questions,
        };

        println!(
            "\nEvaluating SuperGPQA: {} questions (zero-shot CoT)",
            questions.len()
        );

        let mut discipline_correct: HashMap<String, usize> = HashMap::new();
        let mut discipline_total: HashMap<String, usize> = HashMap::new();
        let mut field_correct: HashMap<String, usize> = HashMap::new();
        let mut field_total: HashMap<String, usize> = HashMap::new();
        let mut subfield_correct: HashMap<String, usize> = HashMap::new();
        let mut subfield_total: HashMap<String, usize> = HashMap::new();
        let mut difficulty_correct: HashMap<String, usize> = HashMap::new();
        let mut difficulty_total: HashMap<String, usize> = HashMap::new();

        for q in &questions {
            let question_text = q.question.clone();
            let mut prompt = format!(
                "The following are multiple choice questions (with answers) about {}. Think step by step and then output the answer in the format of \"The answer is (X)\" at the end.\n\n",
                q.subfield
            );
            prompt.push_str(&format!("Question: {}\nOptions: ", question_text));
            for (i, opt) in q.options.iter().enumerate() {
                let letter = (b'A' + i as u8) as char;
                prompt.push_str(&format!("{}: {}\n", letter, opt));
            }
            prompt.push_str("Answer: ");

            let (response, output_tokens, thinking_tokens) =
                client.chat_completion(&model.model_name, "", &prompt)?;
            total_output_tokens += output_tokens.unwrap_or(0);
            total_thinking_tokens += thinking_tokens.unwrap_or(0);

            let pred = extract_answer(&response).ok_or_else(|| {
                eprintln!("  Error extracting answer from: {}", response);
                anyhow::anyhow!("Cannot extract answer")
            })?;

            let is_correct = pred == q.answer_letter.chars().next().unwrap_or(pred);
            if is_correct {
                *discipline_correct.entry(q.discipline.clone()).or_insert(0) += 1;
                *field_correct.entry(q.field.clone()).or_insert(0) += 1;
                *subfield_correct.entry(q.subfield.clone()).or_insert(0) += 1;
                *difficulty_correct.entry(q.difficulty.clone()).or_insert(0) += 1;
            }
            *discipline_total.entry(q.discipline.clone()).or_insert(0) += 1;
            *field_total.entry(q.field.clone()).or_insert(0) += 1;
            *subfield_total.entry(q.subfield.clone()).or_insert(0) += 1;
            *difficulty_total.entry(q.difficulty.clone()).or_insert(0) += 1;

            total_questions += 1;
        }

        // Build record maps for discipline, field, subfield, and difficulty
        fn build_category_record(
            category_correct: &HashMap<String, usize>,
            category_total: &HashMap<String, usize>,
        ) -> serde_json::Map<String, serde_json::Value> {
            let mut record = serde_json::Map::new();
            let all_keys: Vec<String> = category_total.keys().cloned().collect();
            for key in &all_keys {
                let correct = *category_correct.get(key).unwrap_or(&0);
                let total = *category_total.get(key).unwrap_or(&0);
                let wrong = total - correct;
                let acc = if total > 0 {
                    correct as f64 / total as f64
                } else {
                    0.0
                };
                let mut obj = serde_json::Map::new();
                obj.insert("acc".to_string(), serde_json::json!(acc));
                obj.insert("corr".to_string(), serde_json::json!(correct));
                obj.insert("wrong".to_string(), serde_json::json!(wrong));
                record.insert(key.clone(), serde_json::Value::Object(obj));
            }
            record
        }

        let total_correct: usize = discipline_correct.values().sum();
        let overall_accuracy = if total_questions > 0 {
            total_correct as f64 / total_questions as f64
        } else {
            0.0
        };

        let raw_json = serde_json::json!({
            "accuracy": overall_accuracy,
            "results_by_discipline": serde_json::Value::Object(build_category_record(&discipline_correct, &discipline_total)),
            "results_by_field": serde_json::Value::Object(build_category_record(&field_correct, &field_total)),
            "results_by_subfield": serde_json::Value::Object(build_category_record(&subfield_correct, &subfield_total)),
            "results_by_difficulty": serde_json::Value::Object(build_category_record(&difficulty_correct, &difficulty_total)),
            "total_questions": total_questions,
            "output_tokens": total_output_tokens,
            "thinking_tokens": total_thinking_tokens,
        });

        Ok(BenchmarkResult {
            scores: BTreeMap::new(),
            breakdowns: BTreeMap::new(),
            error_classification: BTreeMap::new(),
            artifacts: vec![],
            diagnostics: vec![],
            raw: raw_json,
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
