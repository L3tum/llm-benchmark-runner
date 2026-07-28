use crate::client::{Client, LogprobEntry, ToolCall};
use anyhow::Result;
use std::collections::HashMap;

/// Wraps a `Client` and automatically accumulates output/thinking tokens from every call.
/// Benchmarks receive this wrapper transparently — they call the same methods as `Client`,
/// and the token counts are accumulated without the benchmark needing to track them.
pub struct TokenTracker {
    inner: Client,
    output_tokens: u64,
    thinking_tokens: u64,
}

impl TokenTracker {
    /// Create a new TokenTracker wrapping the given Client.
    pub fn new(inner: Client) -> Self {
        Self {
            inner,
            output_tokens: 0,
            thinking_tokens: 0,
        }
    }

    /// Returns the accumulated output tokens.
    #[must_use]
    pub fn output_tokens(&self) -> u64 {
        self.output_tokens
    }

    /// Returns the accumulated thinking tokens.
    #[must_use]
    pub fn thinking_tokens(&self) -> u64 {
        self.thinking_tokens
    }

    /// Returns the current snapshot of accumulated tokens.
    #[must_use]
    pub fn snapshot(&self) -> (u64, u64) {
        (self.output_tokens, self.thinking_tokens)
    }

    /// Simple chat completion, accumulating tokens.
    pub fn chat_completion(
        &mut self,
        model_name: &str,
        system: &str,
        user: &str,
    ) -> Result<String> {
        let (text, output_tokens, thinking_tokens) =
            self.inner.chat_completion(model_name, system, user)?;
        self.output_tokens += output_tokens.unwrap_or(0);
        self.thinking_tokens += thinking_tokens.unwrap_or(0);
        Ok(text)
    }

    /// Chat completion with structured tool calling and optional conversation history.
    /// Returns the text content and any structured tool calls.
    pub fn chat_completion_with_tools(
        &mut self,
        model_name: &str,
        system: &str,
        user: &str,
        tools: Vec<serde_json::Value>,
        model_params: Option<&HashMap<String, serde_json::Value>>,
        use_history: bool,
    ) -> Result<(String, Vec<ToolCall>)> {
        let (text, tool_calls, output_tokens, thinking_tokens) =
            self.inner.chat_completion_with_tools(
                model_name,
                system,
                user,
                tools,
                model_params,
                use_history,
            )?;
        self.output_tokens += output_tokens.unwrap_or(0);
        self.thinking_tokens += thinking_tokens.unwrap_or(0);
        Ok((text, tool_calls))
    }

    /// Append a tool result to the conversation history.
    pub fn append_tool_result(&mut self, tool_call_id: &str, content: &str) {
        self.inner.append_tool_result(tool_call_id, content);
    }

    /// Clear accumulated conversation history.
    pub fn clear_history(&mut self) {
        self.inner.clear_history();
    }

    /// Get the underlying Client (for health checks etc.).
    #[must_use]
    pub fn inner(&self) -> &Client {
        &self.inner
    }

    /// Chat completion with logprobs, accumulating tokens.
    /// Token counts are tracked internally; callers should use `snapshot()` for totals.
    pub fn chat_completion_logprobs_with_usage(
        &mut self,
        model_name: &str,
        system: &str,
        user: &str,
    ) -> Result<Vec<LogprobEntry>> {
        let (logprobs, output_tokens, thinking_tokens) = self
            .inner
            .chat_completion_logprobs_with_usage(model_name, system, user)?;
        self.output_tokens += output_tokens.unwrap_or(0);
        self.thinking_tokens += thinking_tokens.unwrap_or(0);
        Ok(logprobs)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn snapshot_returns_current_counts() {
        let tracker = TokenTracker {
            inner: Client::new("http://localhost:8080/v1")
                .unwrap_or_else(|_| panic!("mock client")),
            output_tokens: 100,
            thinking_tokens: 30,
        };
        let (out, thinking) = tracker.snapshot();
        assert_eq!(out, 100);
        assert_eq!(thinking, 30);
    }

    #[test]
    fn output_tokens_returns_count() {
        let tracker = TokenTracker {
            inner: Client::new("http://localhost:8080/v1")
                .unwrap_or_else(|_| panic!("mock client")),
            output_tokens: 77,
            thinking_tokens: 0,
        };
        assert_eq!(tracker.output_tokens(), 77);
    }

    #[test]
    fn thinking_tokens_returns_count() {
        let tracker = TokenTracker {
            inner: Client::new("http://localhost:8080/v1")
                .unwrap_or_else(|_| panic!("mock client")),
            output_tokens: 0,
            thinking_tokens: 15,
        };
        assert_eq!(tracker.thinking_tokens(), 15);
    }
}
