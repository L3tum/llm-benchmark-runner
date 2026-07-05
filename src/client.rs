use anyhow::Result;
use serde::{Deserialize, Serialize};
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
    content: String,
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
}
#[derive(Debug, Deserialize)]
struct ModelsResponse {
    data: Vec<serde_json::Value>,
}
pub struct Client {
    base_url: reqwest::Url,
    http: reqwest::blocking::Client,
}
impl Client {
    pub fn new(base_url: &str) -> Result<Self> {
        let base_url = reqwest::Url::parse(base_url)?;
        let http = reqwest::blocking::Client::builder()
            .timeout(std::time::Duration::from_secs(120))
            .build()?;
        Ok(Self { base_url, http })
    }
    pub fn check_health(&self) -> Result<()> {
        let url = self.base_url.join("models")?;
        let resp = self.http.get(url).send()?;
        let response: ModelsResponse = resp.json()?;
        if response.data.is_empty() {
            return Err(anyhow::anyhow!("No models available"));
        }
        Ok(())
    }
    pub fn chat_completion(&self, model_name: &str, system: &str, user: &str) -> Result<String> {
        let url = self.base_url.join("chat/completions")?;
        let req = serde_json::json!({
            "model": model_name,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        });
        let resp = self.http.post(url).json(&req).send()?;
        if !resp.status().is_success() {
            return Err(anyhow::anyhow!("API error: {}", resp.status()));
        }
        let response: ChatResponse = resp.json()?;
        if response.choices.is_empty() {
            return Err(anyhow::anyhow!("Empty response"));
        }
        Ok(response.choices[0].message.content.clone())
    }
    pub fn chat_completion_logprobs(&self, model_name: &str, system: &str, user: &str) -> Result<Vec<LogprobEntry>> {
        let url = self.base_url.join("chat/completions")?;
        let req = serde_json::json!({
            "model": model_name,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "logprobs": true,
            "top_logprobs": 10,
        });
        let resp = self.http.post(url).json(&req).send()?;
        if !resp.status().is_success() {
            return Err(anyhow::anyhow!("API error: {}", resp.status()));
        }
        let response: ChatResponse = resp.json()?;
        if response.choices.is_empty() {
            return Err(anyhow::anyhow!("Empty response"));
        }
        if let Some(logprobs) = &response.choices[0].logprobs {
            if let Some(content) = &logprobs.content {
                if let Some(first) = content.first() {
                    return Ok(first.top_logprobs.clone());
                }
            }
        }
        Err(anyhow::anyhow!("No logprobs"))
    }
}
