use crate::client::Client;
use crate::config::Model;
use anyhow::Result;
use regex::Regex;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fs;
use std::path::PathBuf;

#[derive(Debug, Deserialize, Serialize, Clone)]
pub struct MmluItem {
    pub question: String,
    pub options: Vec<String>,
    pub cot_content: Option<String>,
    pub answer: String,
    pub category: String,
}

pub struct MmluProBenchmark;

impl MmluProBenchmark {
    pub fn download_dataset(&self, split: &str) -> Result<PathBuf> {
        let cache_dir = dirs::cache_dir()
            .unwrap_or_default()
            .join("llm-benchmark-runner")
            .join("mmlu_pro");
        fs::create_dir_all(&cache_dir)?;
        let path = cache_dir.join(format!("{}.json", split));
        if path.exists() {
            return Ok(path);
        }
        let url = format!(
            "https://huggingface.co/datasets/TIGER-Lab/MMLU-Pro/resolve/main/{}/{}.json",
            split, split
        );
        println!("  Downloading MMLU-Pro {} data...", split);
        let response = reqwest::blocking::get(url)?.bytes()?;
        fs::write(&path, response)?;
        Ok(path)
    }

    pub fn load_dataset(&self, path: &PathBuf) -> Result<Vec<MmluItem>> {
        let content = fs::read_to_string(path)?;
        let items: Vec<MmluItem> = serde_json::from_str(&content)?;
        Ok(items)
    }

    fn group_by_category(items: Vec<MmluItem>) -> HashMap<String, Vec<MmluItem>> {
        let mut groups: HashMap<String, Vec<MmluItem>> = HashMap::new();
        for item in items {
            let options: Vec<String> = item.options.into_iter().filter(|o| o != "N/A").collect();
            let category = item.category.clone();
            groups
                .entry(category)
                .or_default()
                .push(MmluItem { options, ..item });
        }
        groups
    }
}

impl super::Benchmark for MmluProBenchmark {
    fn name(&self) -> &str {
        "mmlu_pro"
    }

    fn pre_execute(&self, _config: &serde_yaml::Value) -> Result<()> {
        // Download both test and validation datasets
        self.download_dataset("test")?;
        self.download_dataset("validation")?;
        Ok(())
    }

    fn execute(&self, model: &Model, config: &serde_yaml::Value) -> Result<serde_json::Value> {
        let client = Client::new(&model.proxy)?;
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

        let test_path = self.download_dataset("test")?;
        let val_path = self.download_dataset("validation")?;
        let test_items = self.load_dataset(&test_path)?;
        let val_items = self.load_dataset(&val_path)?;
        let test_data = Self::group_by_category(test_items);
        let val_data = Self::group_by_category(val_items);

        let subjects_to_eval = subjects.unwrap_or_else(|| test_data.keys().cloned().collect());

        let mut category_record: HashMap<String, serde_json::Value> = HashMap::new();
        let mut total_questions = 0usize;
        let mut total_output_tokens: u64 = 0;
        let mut total_thinking_tokens: u64 = 0;

        let choice_map = "ABCDEFGHIJ";

        for category in &subjects_to_eval {
            let test_questions = test_data
                .get(category)
                .ok_or_else(|| anyhow::anyhow!("Category {} not found", category))?
                .clone();
            let test_questions = match num_samples {
                Some(n) if test_questions.len() > n as usize => {
                    test_questions[..n as usize].to_vec()
                }
                _ => test_questions,
            };

            // Few-shot examples from validation set
            let cot_examples: Vec<&MmluItem> = val_data
                .get(category)
                .map(|items| items.iter().take(5).collect())
                .unwrap_or_default();

            println!(
                "\nEvaluating {}: {} questions",
                category,
                test_questions.len()
            );

            let mut category_correct = 0usize;
            let mut category_total = 0usize;

            for q in &test_questions {
                let question_text = q.question.clone();

                // Build prompt with CoT examples
                let mut prompt = format!(
                    "The following are multiple choice questions (with answers) about {}. Think step by step and then output the answer in the format of \"The answer is (X)\" at the end.\n\n",
                    category
                );

                for ex in &cot_examples {
                    prompt.push_str(&format!("Question: {}\nOptions: ", ex.question));
                    for (i, opt) in ex.options.iter().enumerate() {
                        prompt.push_str(&format!("{}: {}\n", &choice_map[i..i + 1], opt));
                    }
                    let cot = ex
                        .cot_content
                        .as_deref()
                        .unwrap_or("Let's think step by step.");
                    let cot_clean = if let Some(stripped) = cot.strip_prefix("A: ") {
                        stripped
                    } else {
                        cot
                    };
                    prompt.push_str(&format!("Answer: {}\n\n", cot_clean));
                }

                // Current question
                prompt.push_str(&format!("Question: {}\nOptions: ", question_text));
                for (i, opt) in q.options.iter().enumerate() {
                    prompt.push_str(&format!("{}: {}\n", &choice_map[i..i + 1], opt));
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
                let is_correct = pred == q.answer.chars().next().unwrap_or(pred);
                if is_correct {
                    category_correct += 1;
                }
                category_total += 1;
                total_questions += 1;
            }

            let accuracy = if category_total > 0 {
                category_correct as f64 / category_total as f64
            } else {
                0.0
            };
            let mut record = serde_json::Map::new();
            record.insert("acc".to_string(), serde_json::json!(accuracy));
            record.insert("corr".to_string(), serde_json::json!(category_correct));
            record.insert(
                "wrong".to_string(),
                serde_json::json!(category_total - category_correct),
            );
            let _: Option<serde_json::Value> =
                category_record.insert(category.clone(), serde_json::Value::Object(record));
        }

        let total_correct: i64 = category_record
            .values()
            .map(|r| r["corr"].as_i64().unwrap_or(0))
            .sum();
        let total_wrong: i64 = category_record
            .values()
            .map(|r| r["wrong"].as_i64().unwrap_or(0))
            .sum();
        let overall_accuracy = if total_correct + total_wrong > 0 {
            total_correct as f64 / (total_correct + total_wrong) as f64
        } else {
            0.0
        };

        Ok(serde_json::json!({
            "accuracy": overall_accuracy,
            "results_by_subject": category_record,
            "total_questions": total_questions,
            "output_tokens": total_output_tokens,
            "thinking_tokens": total_thinking_tokens,
        }))
    }
}

fn extract_answer(text: &str) -> Option<char> {
    // Scan entire text for answer patterns, use the last match
    let re1 = Regex::new(r"\banswer is\s*\(?([A-J])\)?").ok()?;
    let last = re1.captures_iter(text).last();
    if let Some(caps) = last {
        if let Some(m) = caps.get(1) {
            return m.as_str().chars().next();
        }
    }

    let re2 = Regex::new(r"[aA]nswer:\s*([A-J])").ok()?;
    let last = re2.captures_iter(text).last();
    if let Some(caps) = last {
        if let Some(m) = caps.get(1) {
            return m.as_str().chars().next();
        }
    }

    // Final: last single letter from A-J, excluding those followed by comma/semicolon and another letter (e.g., "A, B")
    let re_letter = Regex::new(r"\b([A-J])\b").ok()?;
    let re_sequence = Regex::new(r"[A-J][\s]*[,;:]\s*[A-J]").ok()?;
    let mut last_letter = None;
    for caps in re_letter.captures_iter(text) {
        if let Some(letter_match) = caps.get(1) {
            let letter_start = letter_match.start();
            // Check if this letter is part of a sequence pattern (e.g., "A, B")
            let context = &text[letter_start..text.len().min(letter_start + 6)];
            if re_sequence.find(context).is_some() {
                // This letter is part of a sequence, skip it
                continue;
            }
            last_letter = letter_match.as_str().chars().next();
        }
    }
    last_letter
}
