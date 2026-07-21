use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LogprobEntry {
    pub token: String,
    pub logprob: f64,
}
#[derive(Debug, Deserialize)]
struct Choice {
    message: Message,
    logprobs: Option<Logprobs>,
}
#[derive(Debug, Deserialize)]
struct Message {
    #[serde(default)]
    content: Option<String>,
    #[serde(default)]
    thinking_content: Option<String>,
    #[serde(default)]
    thinking: Option<serde_json::Value>,
}
#[derive(Debug, Deserialize)]
struct Logprobs {
    content: Option<Vec<TokenLogprobs>>,
}
#[derive(Debug, Deserialize)]
struct TokenLogprobs {
    top_logprobs: Vec<LogprobEntry>,
}
#[derive(Debug, Deserialize)]
struct ChatResponse {
    choices: Vec<Choice>,
    usage: Option<Usage>,
}
#[derive(Debug, Deserialize)]
struct Usage {
    completion_tokens: Option<u64>,
    output_tokens: Option<u64>,
    thinking_tokens: Option<u64>,
    completion_tokens_details: Option<TokenDetails>,
    output_tokens_details: Option<TokenDetails>,
}
#[derive(Debug, Deserialize)]
struct TokenDetails {
    reasoning_tokens: Option<u64>,
    thinking_tokens: Option<u64>,
}
#[derive(Debug, Deserialize)]
struct ModelsResponse {
    data: Vec<serde_json::Value>,
}
pub struct Client {
    base_url: reqwest::Url,
    http: reqwest::blocking::Client,
    model_params: Option<serde_json::Map<String, serde_json::Value>>,
}

fn rough_token_count(text: &str) -> u64 {
    text.split_whitespace().count() as u64
}

fn token_usage_from_response(
    response: &ChatResponse,
    message: &Message,
) -> (Option<u64>, Option<u64>) {
    let output_tokens = response
        .usage
        .as_ref()
        .and_then(|usage| usage.completion_tokens.or(usage.output_tokens));
    let thinking_tokens = response
        .usage
        .as_ref()
        .and_then(|usage| usage.thinking_tokens)
        .or_else(|| {
            response.usage.as_ref().and_then(|usage| {
                usage
                    .completion_tokens_details
                    .as_ref()
                    .and_then(|details| details.reasoning_tokens.or(details.thinking_tokens))
            })
        })
        .or_else(|| {
            response.usage.as_ref().and_then(|usage| {
                usage
                    .output_tokens_details
                    .as_ref()
                    .and_then(|details| details.reasoning_tokens.or(details.thinking_tokens))
            })
        })
        .or_else(|| message.thinking_content.as_deref().map(rough_token_count))
        .or_else(|| {
            message
                .thinking
                .as_ref()
                .map(|v| rough_token_count(&v.to_string()))
        });

    (output_tokens, thinking_tokens)
}
impl Client {
    /// Create a client with optional model-level parameters.
    pub fn new_with_model_params(
        base_url: &str,
        model_params: Option<&HashMap<String, serde_json::Value>>,
    ) -> Result<Self> {
        let base_url = reqwest::Url::parse(base_url)?;
        let http = reqwest::blocking::Client::builder()
            .timeout(std::time::Duration::from_secs(120))
            .build()?;
        let model_params = model_params.map(|m| {
            m.iter()
                .map(|(k, v)| (k.clone(), v.clone()))
                .collect::<serde_json::Map<String, serde_json::Value>>()
        });
        Ok(Self {
            base_url,
            http,
            model_params,
        })
    }

    /// Create a client without model-level parameters (thin wrapper).
    pub fn new(base_url: &str) -> Result<Self> {
        Self::new_with_model_params(base_url, None)
    }
    pub fn check_health(&self) -> Result<()> {
        let url = self.base_url.join("models")?;
        let resp = self.http.get(url).send()?.error_for_status()?;
        let response: ModelsResponse = resp.json()?;
        if response.data.is_empty() {
            return Err(anyhow::anyhow!("No models available"));
        }
        Ok(())
    }
    pub fn chat_completion(
        &self,
        model_name: &str,
        system: &str,
        user: &str,
    ) -> Result<(String, Option<u64>, Option<u64>)> {
        let url = self.base_url.join("chat/completions")?;
        let mut req = serde_json::json!({
            "model": model_name,
            "max_tokens": 16_000,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        });
        // Merge model params last so they take final precedence over everything else
        if let Some(params) = &self.model_params {
            if let Some(req_obj) = req.as_object_mut() {
                for (k, v) in params {
                    req_obj.insert(k.clone(), v.clone());
                }
            }
        }
        let resp = self.http.post(url).json(&req).send()?;
        if !resp.status().is_success() {
            return Err(anyhow::anyhow!("API error: {}", resp.status()));
        }
        let response: ChatResponse = resp.json()?;
        if response.choices.is_empty() {
            return Err(anyhow::anyhow!("Empty response"));
        }

        let message = &response.choices[0].message;
        let text = message.content.clone().unwrap_or_default();
        let (output_tokens, thinking_tokens) = token_usage_from_response(&response, message);

        Ok((text, output_tokens, thinking_tokens))
    }
    #[expect(dead_code)]
    pub fn chat_completion_logprobs(
        &self,
        model_name: &str,
        system: &str,
        user: &str,
    ) -> Result<Vec<LogprobEntry>> {
        self.chat_completion_logprobs_with_usage(model_name, system, user)
            .map(|(logprobs, _, _)| logprobs)
    }

    pub fn chat_completion_logprobs_with_usage(
        &self,
        model_name: &str,
        system: &str,
        user: &str,
    ) -> Result<(Vec<LogprobEntry>, Option<u64>, Option<u64>)> {
        let url = self.base_url.join("chat/completions")?;
        let mut req = serde_json::json!({
            "model": model_name,
            "max_tokens": 16_000,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "logprobs": true,
            "top_logprobs": 10,
        });
        // Merge model params last so they take final precedence (overrides max_tokens, logprobs, top_logprobs, etc.)
        if let Some(params) = &self.model_params {
            if let Some(req_obj) = req.as_object_mut() {
                for (k, v) in params {
                    req_obj.insert(k.clone(), v.clone());
                }
            }
        }
        let resp = self.http.post(url).json(&req).send()?;
        if !resp.status().is_success() {
            return Err(anyhow::anyhow!("API error: {}", resp.status()));
        }
        let response: ChatResponse = resp.json()?;
        if response.choices.is_empty() {
            return Err(anyhow::anyhow!("Empty response"));
        }
        let message = &response.choices[0].message;
        let (output_tokens, thinking_tokens) = token_usage_from_response(&response, message);
        if let Some(logprobs) = &response.choices[0].logprobs {
            if let Some(content) = &logprobs.content {
                if let Some(first) = content.first() {
                    return Ok((first.top_logprobs.clone(), output_tokens, thinking_tokens));
                }
            }
        }
        Err(anyhow::anyhow!("No logprobs"))
    }
}
