use crate::client::Client;
use crate::config::Model;
use anyhow::Result;
use regex::Regex;
use std::collections::HashMap;
use std::fs;
use std::path::PathBuf;

/// Single AIME problem with problem statement and integer answer.
#[derive(Debug, Clone, serde::Deserialize)]
pub struct AimeItem {
    pub problem: String,
    pub answer: String,
}

pub struct AimeBenchmark;

fn load_aime_json(path: &PathBuf) -> Result<Vec<AimeItem>> {
    let content = fs::read_to_string(path)?;
    // The JSON file contains an array of objects with `problem` and `answer` fields.
    // The `answer` can be an integer or a string; we'll accept either.
    let items: Vec<serde_json::Value> = serde_json::from_str(&content)?;
    let mut result = Vec::new();
    for item in items {
        let problem = item["problem"].as_str().unwrap_or("").to_string();
        let answer = if let Some(s) = item["answer"].as_str() {
            s.to_string()
        } else if let Some(n) = item["answer"].as_i64() {
            n.to_string()
        } else {
            item["answer"]
                .as_u64()
                .map(|n| n.to_string())
                .unwrap_or_default()
        };
        result.push(AimeItem { problem, answer });
    }
    Ok(result)
}

impl super::Benchmark for AimeBenchmark {
    fn name(&self) -> &str {
        "aime"
    }

    fn pre_execute(&self, config: &serde_yaml::Value) -> Result<()> {
        // Download the AIME 2025 test split
        let year = config
            .get("year")
            .and_then(|v| v.as_str())
            .unwrap_or("2025");
        self.download_dataset(year)?;
        Ok(())
    }

    fn execute(&self, model: &Model, config: &serde_yaml::Value) -> Result<serde_json::Value> {
        let client = Client::new(&model.proxy)?;
        let num_samples: Option<i64> = config.get("num_samples").and_then(|v| v.as_i64());

        let year = config
            .get("year")
            .and_then(|v| v.as_str())
            .unwrap_or("2025");
        let data_path = self.download_dataset(year)?;
        let all_items = load_aime_json(&data_path)?;

        let questions = match num_samples {
            Some(n) if all_items.len() > n as usize => all_items[..n as usize].to_vec(),
            _ => all_items,
        };

        println!(
            "\nEvaluating AIME {}: {} problems (zero-shot CoT)",
            year,
            questions.len()
        );

        let mut correct = 0usize;
        let mut total = 0usize;
        let mut total_output_tokens: u64 = 0;
        let mut total_thinking_tokens: u64 = 0;
        let mut problem_results: HashMap<usize, serde_json::Value> = HashMap::new();

        for (idx, q) in questions.iter().enumerate() {
            let problem_text = q.problem.clone();
            let prompt = format!(
                "You are a math competition solver. Solve the following problem step by step. The answer is an integer between 000 and 999. Put your final answer in the format of \"\\boxed{{answer}}\" at the end.\n\n{}\nPlease reason step by step, and put your final answer within \\boxed{{}}.",
                problem_text
            );

            let (response, output_tokens, thinking_tokens) =
                client.chat_completion(&model.model_name, "", &prompt)?;
            total_output_tokens += output_tokens.unwrap_or(0);
            total_thinking_tokens += thinking_tokens.unwrap_or(0);
            let extracted_answer = extract_int_answer(&response);
            let is_correct = extracted_answer.as_deref() == Some(&q.answer);
            if is_correct {
                correct += 1;
            }
            total += 1;

            problem_results.insert(
                idx,
                serde_json::json!({
                    "correct": is_correct,
                    "expected": q.answer,
                    "extracted": extracted_answer.unwrap_or_default()
                }),
            );
        }

        let accuracy = if total > 0 {
            correct as f64 / total as f64
        } else {
            0.0
        };

        Ok(serde_json::json!({
            "accuracy": accuracy,
            "total_questions": total,
            "correct": correct,
            "problem_results": problem_results,
            "output_tokens": total_output_tokens,
            "thinking_tokens": total_thinking_tokens,
        }))
    }
}

impl AimeBenchmark {
    /// Download AIME dataset JSON from HuggingFace via the MathArena datasets viewer API.
    /// Supports `year` parameter to switch between AIME 2025 and 2026 datasets.
    pub fn download_dataset(&self, year: &str) -> Result<PathBuf> {
        let cache_dir = dirs::cache_dir()
            .unwrap_or_default()
            .join("llm-benchmark-runner")
            .join("aime");
        fs::create_dir_all(&cache_dir)?;
        let path = cache_dir.join(format!("aime_{}.json", year));
        if path.exists() {
            return Ok(path);
        }

        // Use the HuggingFace datasets viewer API to fetch all rows as JSON
        // MathArena/aime_2025 and MathArena/aime_2026 have train split with 30 rows
        let url = format!(
            "https://datasets-server.huggingface.co/rows?dataset=MathArena/aime_{0}&config=default&split=train&offset=0&length=100",
            year
        );
        println!(
            "  Downloading AIME {} data from MathArena via datasets viewer API...",
            year
        );

        let response = reqwest::blocking::get(url)?;
        if !response.status().is_success() {
            return Err(anyhow::anyhow!(
                "Failed to download AIME {} dataset (HTTP {}): {:?}",
                year,
                response.status(),
                response.text()
            ));
        }
        let api_result: serde_json::Value = response.json()?;
        // Extract rows from the API response and save as a JSON array
        let rows: Vec<serde_json::Value> = api_result["rows"]
            .as_array()
            .ok_or_else(|| anyhow::anyhow!("Invalid API response"))?
            .iter()
            .map(|row| row["row"].clone())
            .collect();
        let json_bytes = serde_json::to_vec_pretty(&rows)?;
        fs::write(&path, json_bytes)?;
        Ok(path)
    }
}

/// Extract a 3-digit integer answer from a model response using regex.
/// Looks for patterns like "\boxed{000}" or "\boxed{123}".
fn extract_int_answer(text: &str) -> Option<String> {
    let re = Regex::new(r"\\boxed\{(\d+)\}")
        .ok()
        .and_then(|r| r.captures_iter(text).last())
        .and_then(|caps| caps.get(1))
        .map(|m| m.as_str().to_string());
    re
}
