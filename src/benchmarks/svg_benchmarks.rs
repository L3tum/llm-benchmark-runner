use crate::benchmarks::Benchmark;
use crate::config;
use crate::config::Model;
use crate::shared::{Artifact, BenchmarkCategory, BenchmarkResult, Score, ScoreUnit, TaskResult};
use crate::token_tracker::TokenTracker;
use anyhow::Result;
use std::collections::BTreeMap;
use std::fs;
use std::path::PathBuf;
use std::sync::Mutex;

/// SVG generation: flamingo moonwalking on a beach.
pub struct SvgMoonwalkBenchmark {
    state: Mutex<SvgState>,
}

/// SVG generation: pelican riding a bike.
pub struct SvgBikeBenchmark {
    state: Mutex<SvgState>,
}

#[derive(Clone, serde::Serialize, serde::Deserialize)]
struct SvgState {
    done: bool,
    output_dir: PathBuf,
}

const MOONWALK_PROMPT: &str = r#"Generate a valid SVG image of a flamingo moonwalking on a beach. The SVG should include:
- A flamingo with pink feathers and long legs
- The flamingo in a moonwalking pose (one leg forward, one back)
- A beach scene with sand and ocean in the background
- A sunset or daytime sky

The SVG must be well-formed XML, use viewBox, and render correctly. Output ONLY the SVG code, nothing else."#;

const BIKE_PROMPT: &str = r#"Generate a valid SVG image of a pelican riding a bicycle. The SVG should include:
- A pelican with a large beak/pouch
- A bicycle with two wheels, handlebars, and frame
- The pelican positioned as if riding the bike
- A simple background (road, sky, or grass)

The SVG must be well-formed XML, use viewBox, and render correctly. Output ONLY the SVG code, nothing else."#;

impl Default for SvgMoonwalkBenchmark {
    fn default() -> Self {
        Self {
            state: Mutex::new(SvgState {
                done: false,
                output_dir: PathBuf::from("output"),
            }),
        }
    }
}

impl Default for SvgBikeBenchmark {
    fn default() -> Self {
        Self {
            state: Mutex::new(SvgState {
                done: false,
                output_dir: PathBuf::from("output"),
            }),
        }
    }
}

impl Benchmark for SvgMoonwalkBenchmark {
    fn name(&self) -> &str {
        "svg_moonwalk"
    }

    fn display_name(&self) -> &'static str {
        "SVG: Flamingo Moonwalking"
    }

    fn category(&self) -> BenchmarkCategory {
        BenchmarkCategory::Creative
    }

    fn pre_execute(&self, config: &yaml_serde::Value) -> Result<()> {
        let output_dir =
            config::extract_string(config, "output_dir").unwrap_or_else(|| "output".to_string());
        {
            let mut state = self.state.lock().unwrap_or_else(|p| p.into_inner());
            state.output_dir = PathBuf::from(output_dir);
            fs::create_dir_all(&state.output_dir)?;
        }
        Ok(())
    }

    fn execute_one(
        &self,
        model: &Model,
        _config: &yaml_serde::Value,
        tracker: &mut TokenTracker,
    ) -> Result<Option<TaskResult>> {
        let should_execute = {
            let mut state = self.state.lock().unwrap_or_else(|p| p.into_inner());
            if state.done {
                return Ok(None);
            }
            state.done = true;
            true
        };

        if !should_execute {
            return Ok(None);
        }

        let response = tracker.chat_completion(&model.model_name, "", MOONWALK_PROMPT)?;

        let svg_content = extract_svg(&response);

        let output_dir = self
            .state
            .lock()
            .expect(crate::shared::MUTEX_PANIC_MSG)
            .output_dir
            .clone();
        let svg_path = output_dir.join("flamingo_moonwalk.svg");
        fs::write(&svg_path, &svg_content)?;

        let is_valid = validate_svg(&svg_content);
        let pass = is_valid;

        Ok(Some(
            TaskResult::new("task-0", pass, if pass { 1.0 } else { 0.0 }, vec![]).with_metadata(
                Some(serde_json::json!({
                    "valid_svg": is_valid,
                    "svg_length": svg_content.len(),
                    "svg_path": svg_path.to_string_lossy().to_string(),
                })),
            ),
        ))
    }

    fn to_report_result(&self, b: &BenchmarkResult) -> Result<BenchmarkResult> {
        let raw = &b.raw;

        let (pass, output_tokens, thinking_tokens, svg_path) = {
            if let Some(per_task) = raw.get("per_task").and_then(|v| v.as_array()) {
                if let Some(task) = per_task.first() {
                    let p = task
                        .get("passed")
                        .and_then(|v| v.as_bool())
                        .unwrap_or(false);
                    let out = task
                        .get("output_tokens")
                        .and_then(|v| v.as_i64())
                        .unwrap_or(0);
                    let think = task
                        .get("thinking_tokens")
                        .and_then(|v| v.as_i64())
                        .unwrap_or(0);
                    let path = task
                        .get("metadata")
                        .and_then(|m| m.get("svg_path"))
                        .and_then(|v| v.as_str())
                        .unwrap_or("")
                        .to_string();
                    (p, out, think, path)
                } else {
                    (false, 0, 0, String::new())
                }
            } else {
                (false, 0, 0, String::new())
            }
        };

        let mut scores = BTreeMap::new();
        scores.insert(
            "valid".to_string(),
            Score::bool(pass).primary(true).higher_is_better(true),
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

        let mut artifacts = vec![];
        if !svg_path.is_empty() {
            artifacts.push(Artifact {
                label: "SVG Output".to_string(),
                path: svg_path,
                kind: "svg".to_string(),
            });
        }

        Ok(BenchmarkResult {
            scores,
            breakdowns: BTreeMap::new(),
            error_classification: BTreeMap::new(),
            artifacts,
            diagnostics: vec![crate::reports::model::Diagnostic {
                level: "info".to_string(),
                message: format!(
                    "SVG Moonwalk: {}",
                    if pass {
                        "valid SVG generated"
                    } else {
                        "invalid or malformed SVG"
                    }
                ),
            }],
            raw: raw.clone(),
        })
    }
}

impl Benchmark for SvgBikeBenchmark {
    fn name(&self) -> &str {
        "svg_bike"
    }

    fn display_name(&self) -> &'static str {
        "SVG: Pelican Riding a Bike"
    }

    fn category(&self) -> BenchmarkCategory {
        BenchmarkCategory::Creative
    }

    fn pre_execute(&self, config: &yaml_serde::Value) -> Result<()> {
        let output_dir =
            config::extract_string(config, "output_dir").unwrap_or_else(|| "output".to_string());
        {
            let mut state = self.state.lock().unwrap_or_else(|p| p.into_inner());
            state.output_dir = PathBuf::from(output_dir);
            fs::create_dir_all(&state.output_dir)?;
        }
        Ok(())
    }

    fn execute_one(
        &self,
        model: &Model,
        _config: &yaml_serde::Value,
        tracker: &mut TokenTracker,
    ) -> Result<Option<TaskResult>> {
        let should_execute = {
            let mut state = self.state.lock().unwrap_or_else(|p| p.into_inner());
            if state.done {
                return Ok(None);
            }
            state.done = true;
            true
        };

        if !should_execute {
            return Ok(None);
        }

        let response = tracker.chat_completion(&model.model_name, "", BIKE_PROMPT)?;

        let svg_content = extract_svg(&response);

        let output_dir = self
            .state
            .lock()
            .expect(crate::shared::MUTEX_PANIC_MSG)
            .output_dir
            .clone();
        let svg_path = output_dir.join("pelican_bike.svg");
        fs::write(&svg_path, &svg_content)?;

        let is_valid = validate_svg(&svg_content);
        let pass = is_valid;

        Ok(Some(
            TaskResult::new("task-0", pass, if pass { 1.0 } else { 0.0 }, vec![]).with_metadata(
                Some(serde_json::json!({
                    "valid_svg": is_valid,
                    "svg_length": svg_content.len(),
                    "svg_path": svg_path.to_string_lossy().to_string(),
                })),
            ),
        ))
    }

    fn to_report_result(&self, b: &BenchmarkResult) -> Result<BenchmarkResult> {
        let raw = &b.raw;

        let (pass, output_tokens, thinking_tokens, svg_path) = {
            if let Some(per_task) = raw.get("per_task").and_then(|v| v.as_array()) {
                if let Some(task) = per_task.first() {
                    let p = task
                        .get("passed")
                        .and_then(|v| v.as_bool())
                        .unwrap_or(false);
                    let out = task
                        .get("output_tokens")
                        .and_then(|v| v.as_i64())
                        .unwrap_or(0);
                    let think = task
                        .get("thinking_tokens")
                        .and_then(|v| v.as_i64())
                        .unwrap_or(0);
                    let path = task
                        .get("metadata")
                        .and_then(|m| m.get("svg_path"))
                        .and_then(|v| v.as_str())
                        .unwrap_or("")
                        .to_string();
                    (p, out, think, path)
                } else {
                    (false, 0, 0, String::new())
                }
            } else {
                (false, 0, 0, String::new())
            }
        };

        let mut scores = BTreeMap::new();
        scores.insert(
            "valid".to_string(),
            Score::bool(pass).primary(true).higher_is_better(true),
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

        let mut artifacts = vec![];
        if !svg_path.is_empty() {
            artifacts.push(Artifact {
                label: "SVG Output".to_string(),
                path: svg_path,
                kind: "svg".to_string(),
            });
        }

        Ok(BenchmarkResult {
            scores,
            breakdowns: BTreeMap::new(),
            error_classification: BTreeMap::new(),
            artifacts,
            diagnostics: vec![crate::reports::model::Diagnostic {
                level: "info".to_string(),
                message: format!(
                    "SVG Bike: {}",
                    if pass {
                        "valid SVG generated"
                    } else {
                        "invalid or malformed SVG"
                    }
                ),
            }],
            raw: raw.clone(),
        })
    }
}

/// Extract SVG content from a response, handling markdown code blocks.
fn extract_svg(response: &str) -> String {
    if let Some(start) = response.find("```") {
        let after_backticks = &response[start + 3..];
        let after_lang = if let Some(newline) = after_backticks.find('\n') {
            &after_backticks[newline + 1..]
        } else {
            after_backticks
        };

        if let Some(end) = after_lang.rfind("```") {
            return after_lang[..end].trim().to_string();
        }
    }

    if let Some(start) = response.find("<svg") {
        let from_svg = &response[start..];
        if let Some(end) = from_svg.find("</svg>") {
            return from_svg[..end + 6].to_string();
        }
    }

    response.trim().to_string()
}

/// Validate that the SVG is well-formed XML using quick-xml.
fn validate_svg(svg: &str) -> bool {
    let trimmed = svg.trim();

    if !trimmed.starts_with("<svg") {
        return false;
    }

    if !trimmed.ends_with("</svg>") {
        return false;
    }

    // Parse with quick-xml reader to check well-formedness
    let mut reader = quick_xml::Reader::from_str(trimmed);
    let mut valid = true;
    let mut depth = 0i32;
    loop {
        match reader.read_event() {
            Ok(quick_xml::events::Event::Start(_) | quick_xml::events::Event::Empty(_)) => {
                depth += 1;
            }
            Ok(quick_xml::events::Event::End(_)) => {
                depth -= 1;
            }
            Ok(quick_xml::events::Event::Eof) => break,
            Ok(_) => {}
            Err(_) => {
                valid = false;
                break;
            }
        }
    }
    valid && depth == 0
}
