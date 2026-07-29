use crate::client::{Client, LogprobEntry, ToolCall};
use anyhow::Result;
use std::collections::HashMap;

/// Wraps a `Client` and automatically accumulates output/thinking tokens from every call.
/// Benchmarks receive this wrapper transparently — they call the same methods as `Client`,
/// and the token counts are accumulated without the benchmark needing to track them.
/// Also tracks tool call success/failure (schema validation) for _tools benchmarks.
pub struct TokenTracker {
    inner: Client,
    output_tokens: u64,
    thinking_tokens: u64,
    /// Total tool calls made across all chat_completion_with_tools calls.
    tool_calls_total: u64,
    /// Tool calls whose name and arguments matched the provided schema.
    tool_calls_valid: u64,
    /// Tool calls that did not match the provided schema.
    tool_calls_invalid: u64,
}

impl TokenTracker {
    /// Create a new TokenTracker wrapping the given Client.
    pub fn new(inner: Client) -> Self {
        Self {
            inner,
            output_tokens: 0,
            thinking_tokens: 0,
            tool_calls_total: 0,
            tool_calls_valid: 0,
            tool_calls_invalid: 0,
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

    /// Returns the total number of tool calls made.
    #[must_use]
    pub fn tool_calls_total(&self) -> u64 {
        self.tool_calls_total
    }

    /// Returns the number of tool calls that matched the schema.
    #[must_use]
    pub fn tool_calls_valid(&self) -> u64 {
        self.tool_calls_valid
    }

    /// Returns the number of tool calls that did not match the schema.
    #[must_use]
    pub fn tool_calls_invalid(&self) -> u64 {
        self.tool_calls_invalid
    }

    /// Validate and record tool calls against the provided schemas.
    /// For each tool call, checks that:
    /// 1. The tool name matches a tool in the schemas
    /// 2. All required parameters are present
    /// 3. Parameter types are compatible (string, number, integer, boolean, array, object)
    pub fn record_tool_calls(&mut self, tool_calls: &[ToolCall], schemas: &[serde_json::Value]) {
        for tc in tool_calls {
            self.tool_calls_total += 1;
            if Self::validate_tool_call(tc, schemas) {
                self.tool_calls_valid += 1;
            } else {
                self.tool_calls_invalid += 1;
            }
        }
    }

    /// Validate a single tool call against the provided schemas.
    fn validate_tool_call(tc: &ToolCall, schemas: &[serde_json::Value]) -> bool {
        // Find the matching tool schema by name
        let Some(schema) = schemas.iter().find(|s| {
            s.get("function")
                .and_then(|f| f.get("name"))
                .and_then(|n| n.as_str())
                .is_some_and(|name| name == tc.name)
        }) else {
            return false; // unknown tool name
        };

        // Get the parameters schema
        let params = schema
            .get("function")
            .and_then(|f| f.get("parameters"))
            .and_then(|p| p.get("properties"));

        let Some(params) = params else {
            return true; // no parameters schema, accept any call
        };

        // Check required fields are present
        let required: Vec<&str> = schema
            .get("function")
            .and_then(|f| f.get("parameters"))
            .and_then(|p| p.get("required"))
            .and_then(|r| r.as_array())
            .map(|arr| arr.iter().filter_map(|v| v.as_str()).collect())
            .unwrap_or_default();

        let Some(args) = tc.arguments.as_object() else {
            return required.is_empty(); // no object args but no required fields
        };

        for req in &required {
            if !args.contains_key(*req) {
                return false;
            }
        }

        // Check type compatibility for each provided argument
        for (key, value) in args {
            if let Some(prop_schema) = params.get(key) {
                if let Some(expected_type) = prop_schema.get("type").and_then(|t| t.as_str()) {
                    if !Self::type_matches(expected_type, value) {
                        return false;
                    }
                }
            }
        }

        true
    }

    /// Check if a JSON value matches an expected JSON Schema type.
    fn type_matches(expected: &str, value: &serde_json::Value) -> bool {
        match expected {
            "string" => value.is_string(),
            "integer" => value.is_number() && value.as_f64().is_some_and(|n| n.fract() == 0.0),
            "number" => value.is_number(),
            "boolean" => value.is_boolean(),
            "array" => value.is_array(),
            "object" => value.is_object(),
            _ => true, // unknown type, accept
        }
    }

    /// Returns the current snapshot of accumulated tokens.
    #[must_use]
    pub fn snapshot(&self) -> (u64, u64) {
        (self.output_tokens, self.thinking_tokens)
    }

    /// Returns the current snapshot of tool call tracking.
    #[must_use]
    pub fn tool_call_snapshot(&self) -> (u64, u64, u64) {
        (
            self.tool_calls_total,
            self.tool_calls_valid,
            self.tool_calls_invalid,
        )
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
    use crate::client::ToolCall;

    #[test]
    fn snapshot_returns_current_counts() {
        let tracker = TokenTracker {
            inner: Client::new("http://localhost:8080/v1")
                .unwrap_or_else(|_| panic!("mock client")),
            output_tokens: 100,
            thinking_tokens: 30,
            tool_calls_total: 0,
            tool_calls_valid: 0,
            tool_calls_invalid: 0,
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
            tool_calls_total: 0,
            tool_calls_valid: 0,
            tool_calls_invalid: 0,
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
            tool_calls_total: 0,
            tool_calls_valid: 0,
            tool_calls_invalid: 0,
        };
        assert_eq!(tracker.thinking_tokens(), 15);
    }

    #[test]
    fn tool_call_snapshot_returns_counts() {
        let tracker = TokenTracker {
            inner: Client::new("http://localhost:8080/v1")
                .unwrap_or_else(|_| panic!("mock client")),
            output_tokens: 0,
            thinking_tokens: 0,
            tool_calls_total: 5,
            tool_calls_valid: 4,
            tool_calls_invalid: 1,
        };
        let (total, valid, invalid) = tracker.tool_call_snapshot();
        assert_eq!(total, 5);
        assert_eq!(valid, 4);
        assert_eq!(invalid, 1);
    }

    #[test]
    fn validate_tool_call_matching_schema() {
        let schema: Vec<serde_json::Value> = serde_json::from_value(serde_json::json!([{
            "type": "function",
            "function": {
                "name": "reverse_string",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "input": { "type": "string" }
                    },
                    "required": ["input"]
                }
            }
        }]))
        .unwrap();
        let tc = ToolCall {
            id: "call_1".to_string(),
            name: "reverse_string".to_string(),
            arguments: serde_json::json!({"input": "hello"}),
        };
        assert!(TokenTracker::validate_tool_call(&tc, &schema));
    }

    #[test]
    fn validate_tool_call_missing_required_field() {
        let schema: Vec<serde_json::Value> = serde_json::from_value(serde_json::json!([{
            "type": "function",
            "function": {
                "name": "reverse_string",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "input": { "type": "string" }
                    },
                    "required": ["input"]
                }
            }
        }]))
        .unwrap();
        let tc = ToolCall {
            id: "call_1".to_string(),
            name: "reverse_string".to_string(),
            arguments: serde_json::json!({}),
        };
        assert!(!TokenTracker::validate_tool_call(&tc, &schema));
    }

    #[test]
    fn validate_tool_call_wrong_type() {
        let schema: Vec<serde_json::Value> = serde_json::from_value(serde_json::json!([{
            "type": "function",
            "function": {
                "name": "add_block",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "x": { "type": "integer" },
                        "name": { "type": "string" }
                    },
                    "required": ["x", "name"]
                }
            }
        }]))
        .unwrap();
        let tc = ToolCall {
            id: "call_1".to_string(),
            name: "add_block".to_string(),
            arguments: serde_json::json!({"x": "not_an_int", "name": "test"}),
        };
        assert!(!TokenTracker::validate_tool_call(&tc, &schema));
    }

    #[test]
    fn validate_tool_call_unknown_tool() {
        let schema: Vec<serde_json::Value> = serde_json::from_value(serde_json::json!([{
            "type": "function",
            "function": {
                "name": "reverse_string",
                "parameters": {}
            }
        }]))
        .unwrap();
        let tc = ToolCall {
            id: "call_1".to_string(),
            name: "unknown_tool".to_string(),
            arguments: serde_json::json!({}),
        };
        assert!(!TokenTracker::validate_tool_call(&tc, &schema));
    }
}
