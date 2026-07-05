use crate::benchmarks::mmlu_pro::MmluProBenchmark;
use crate::client::{Client, LogprobEntry};
use crate::config::Model;
use anyhow::Result;
use std::collections::HashMap;
use std::fs;

fn load_prompts_from_file(path: &str, num_prompts: usize) -> Result<Vec<String>> {
    let content = fs::read_to_string(path)?;
    // Try JSON array first
    if let Ok(strings) = serde_json::from_str::<Vec<String>>(&content) {
        Ok(strings.into_iter().take(num_prompts).collect())
    } else {
        // Fallback: newline-separated text, skip empty lines
        let prompts: Vec<String> = content
            .lines()
            .filter(|line| !line.trim().is_empty())
            .take(num_prompts)
            .map(|s| s.to_string())
            .collect();
        Ok(prompts)
    }
}

pub struct KldBenchmark;

fn compute_kl_from_logprobs(logprobs_a: &[LogprobEntry], logprobs_b: &[LogprobEntry]) -> f64 {
    if logprobs_a.is_empty() || logprobs_b.is_empty() {
        return f64::INFINITY;
    }

    // Convert to probability distributions over common tokens
    fn logprobs_to_dist(lp: &[LogprobEntry]) -> HashMap<&str, f64> {
        let mut dist: HashMap<&str, f64> = HashMap::new();
        let max_logprob = lp.iter().map(|e| e.logprob).fold(f64::NEG_INFINITY, f64::max);
        for entry in lp {
            dist.insert(
                &entry.token,
                (entry.logprob - max_logprob).exp(),
            );
        }
        let total: f64 = dist.values().sum();
        if total == 0.0 {
            return HashMap::new();
        }
        for val in dist.values_mut() {
            *val /= total;
        }
        dist
    }

    let dist_a = logprobs_to_dist(logprobs_a);
    let dist_b = logprobs_to_dist(logprobs_b);

    let mut kl = 0.0;
    for (&token, p) in &dist_a {
        let q = *dist_b.get(token).unwrap_or(&0.0);
        let p = *p;
        if p > 0.0 {
            if q > 0.0 {
                kl += p * (p / q).ln();
            } else {
                return f64::INFINITY;
            }
        }
    }
    kl
}

impl super::Benchmark for KldBenchmark {
    fn name(&self) -> &str {
        "kld"
    }

    fn pre_execute(&self, _config: &serde_yaml::Value) -> Result<()> {
        let mmlu = MmluProBenchmark;
        mmlu.pre_execute(&serde_yaml::Value::Null)?;
        Ok(())
    }

    fn execute(&self, model: &Model, config: &serde_yaml::Value) -> Result<serde_json::Value> {
        let client = Client::new(&model.proxy)?;
        let num_prompts: usize = config
            .get("num_prompts")
            .and_then(|v| v.as_i64())
            .unwrap_or(10) as usize;
        let prompt_source = config
            .get("prompt_source")
            .and_then(|v| v.as_str())
            .unwrap_or("mmlu");
        let custom_prompts_path = config.get("custom_prompts_path").and_then(|v| v.as_str());

        let prompts: Vec<String> = match custom_prompts_path {
            Some(path) => {
                println!("  Loading custom prompts from {}", path);
                load_prompts_from_file(path, num_prompts)?
            }
            None if prompt_source == "mmlu" => {
                println!("  Using MMLU-Pro test prompts for KLD");
                let mmlu = MmluProBenchmark;
                let test_path = mmlu.download_dataset("test")?;
                let items = mmlu.load_dataset(&test_path)?;
                items
                    .into_iter()
                    .take(num_prompts)
                    .map(|i| i.question)
                    .collect()
            }
            None => {
                return Err(anyhow::anyhow!(
                    "prompt_source '{}' not implemented; use 'mmlu' or set custom_prompts_path",
                    prompt_source
                ))
            }
        };

        if prompts.is_empty() {
            return Err(anyhow::anyhow!("No prompts loaded for KLD"));
        }

        let mut model_logprobs: Vec<Vec<LogprobEntry>> = Vec::new();
        let mut failed = 0;

        for prompt in &prompts {
            match client.chat_completion_logprobs(&model.model_name, "You are a helpful assistant.", prompt) {
                Ok(logprobs) => {
                    model_logprobs.push(logprobs);
                }
                Err(e) => {
                    eprintln!("  Error getting logprobs for {}: {}", model.display_name, e);
                    model_logprobs.push(Vec::new());
                    failed += 1;
                }
            }
        }

        if failed > prompts.len() as i64 * 3 / 10 {
            println!(
                "  WARNING: {}/{} KLD prompt failures ({:.0}%%)",
                failed,
                prompts.len(),
                (failed as f64 / prompts.len() as f64) * 100.0
            );
        }

        Ok(serde_json::json!({
            "model": model.display_name,
            "num_prompts": prompts.len(),
            "kld": model_logprobs,
        }))
    }

    fn post_execute(
        &self,
        model_results: &HashMap<String, serde_json::Value>,
    ) -> Result<serde_json::Value> {
        let mut all_logits: HashMap<String, Vec<Vec<LogprobEntry>>> = HashMap::new();
        for (name, data) in model_results {
            if let Some(kld_arr) = data.get("kld").and_then(|v| v.as_array()) {
                let mut entries: Vec<Vec<LogprobEntry>> = Vec::new();
                for arr in kld_arr {
                    if let Some(inner) = arr.as_array() {
                        let mut logprobs: Vec<LogprobEntry> = Vec::new();
                        for item in inner {
                            if let (Some(token), Some(logprob)) =
                                (item.get("token"), item.get("logprob"))
                            {
                                if let (Some(token), Some(logprob)) =
                                    (token.as_str(), logprob.as_f64())
                                {
                                    logprobs.push(LogprobEntry {
                    token: token.to_string(),
                                        logprob,
                                    });
                                }
                            }
                        }
                        entries.push(logprobs);
                    }
                }
                all_logits.insert(name.clone(), entries);
            }
        }

        let names: Vec<String> = all_logits.keys().cloned().collect();
        let mut pairwise = serde_json::Map::new();
        let mut kld_pairs: HashMap<(&str, &str), Vec<f64>> = HashMap::new();

        for (i, a_name) in names.iter().enumerate() {
            for b_name in names[i + 1..].iter() {
                let logits_a = &all_logits[a_name];
                let logits_b = &all_logits[b_name];
                let len = std::cmp::min(logits_a.len(), logits_b.len());
                let mut kld_values: Vec<f64> = Vec::new();
                for (logprobs_a, logprobs_b) in logits_a.iter().zip(logits_b.iter()).take(len) {
                    let kl = compute_kl_from_logprobs(logprobs_a, logprobs_b);
                    if kl.is_finite() {
                        kld_values.push(kl);
                    }
                }
                if !kld_values.is_empty() {
                    let avg_kld: f64 = kld_values.iter().sum::<f64>() / kld_values.len() as f64;
                    let key = format!("{}_vs_{}", a_name, b_name);
                    pairwise.insert(
                        key.clone(),
                        serde_json::json!({
                            "models": [a_name, b_name],
                            "avg_kld": avg_kld,
                            "num_prompts_evaluated": kld_values.len(),
                            "kld_values": kld_values,
                        }),
                    );
                    kld_pairs
                        .insert((a_name.as_str(), b_name.as_str()), kld_values.clone());
                    kld_pairs.insert((b_name.as_str(), a_name.as_str()), kld_values);
                }
            }
        }

        // Compute average KLD to all other models
        let mut avg_kld_to_others = serde_json::Map::new();
        for model_name in &names {
            let mut kld_to_others = Vec::new();
            for other in &names {
                if other == model_name {
                    continue;
                }
                if let Some(values) = kld_pairs.get(&(model_name.as_str(), other.as_str())) {
                    let avg = values.iter().sum::<f64>() / values.len() as f64;
                    kld_to_others.push(avg);
                }
            }
            if !kld_to_others.is_empty() {
                let overall_avg = kld_to_others.iter().sum::<f64>() / kld_to_others.len() as f64;
                avg_kld_to_others.insert(
                    model_name.clone(),
                    serde_json::json!({
                        "avg_kld_to_others": overall_avg,
                        "klds": kld_to_others,
                    }),
                );
            }
        }
        pairwise.insert("avg_kld_to_others".to_string(), serde_json::Value::Object(avg_kld_to_others));
        Ok(serde_json::json!(pairwise))
    }
}
