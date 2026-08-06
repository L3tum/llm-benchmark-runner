use crate::benchmarks::Benchmark;
use crate::config::{self, Model};
use crate::docker_runner::{DockerRunConfig, DockerRunner};
use crate::shared::{
    BenchmarkCategory, BenchmarkResult, BreakdownTable, Score, ScoreUnit, TaskResult,
};
use crate::token_tracker::TokenTracker;
use anyhow::{Context, Result};
use serde::Deserialize;
use std::collections::BTreeMap;
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::Mutex;

pub struct TerminalBenchBenchmark {
    /// Internal iteration state protected by Mutex for execute_one
    state: Mutex<TerminalBenchState>,
}

struct TerminalBenchState {
    tasks: Vec<TerminalBenchTask>,
    current_idx: usize,
    config: TerminalBenchConfig,
}

#[derive(Debug, Clone, Deserialize)]
#[allow(dead_code)] // task field kept for schema alignment
struct TerminalBenchTask {
    #[serde(flatten)]
    metadata: TaskMetadata,
    #[serde(default)]
    task: Option<TaskInfo>,
    #[serde(default)]
    environment: Option<EnvConfig>,
}

#[derive(Debug, Clone, Deserialize)]
struct TaskMetadata {
    #[serde(rename = "task")]
    task_info: Option<TaskInfo>,
    #[serde(default)]
    metadata: Option<TaskDetails>,
}

#[derive(Debug, Clone, Deserialize)]
#[allow(dead_code)] // description and keywords kept for schema alignment
struct TaskInfo {
    #[serde(default)]
    name: String,
    #[serde(default)]
    description: String,
    #[serde(default)]
    keywords: Vec<String>,
}

#[derive(Debug, Clone, Deserialize)]
#[allow(dead_code)] // tags field kept for schema alignment
struct TaskDetails {
    #[serde(default = "default_difficulty")]
    difficulty: String,
    #[serde(default = "default_category")]
    category: String,
    #[serde(default)]
    #[allow(dead_code)] // kept for schema completeness
    tags: Vec<String>,
}

fn default_difficulty() -> String {
    "medium".to_string()
}

fn default_category() -> String {
    "other".to_string()
}

#[derive(Debug, Clone, Deserialize)]
#[allow(dead_code)] // resource fields kept for schema alignment
struct EnvConfig {
    #[serde(default)]
    docker_image: String,
    #[serde(default)]
    #[allow(dead_code)] // kept for schema completeness
    cpus: u64,
    #[serde(default = "default_memory")]
    memory_mb: u64,
    #[serde(default = "default_storage")]
    #[allow(dead_code)] // kept for schema completeness
    storage_mb: u64,
    #[serde(default)]
    #[allow(dead_code)] // kept for schema completeness
    gpus: u64,
    #[serde(default)]
    allow_internet: bool,
}

fn default_memory() -> u64 {
    4096
}

fn default_storage() -> u64 {
    10240
}

#[derive(Debug, Clone)]
#[allow(dead_code)] // num_samples and categories reserved for future filtering
struct TerminalBenchConfig {
    #[allow(dead_code)] // kept for future sampling support
    num_samples: Option<usize>,
    max_iterations: usize,
    timeout_secs: u64,
    #[allow(dead_code)] // kept for future filtering support
    categories: Option<Vec<String>>,
}

impl Default for TerminalBenchBenchmark {
    fn default() -> Self {
        Self {
            state: Mutex::new(TerminalBenchState {
                tasks: Vec::new(),
                current_idx: 0,
                config: TerminalBenchConfig {
                    num_samples: None,
                    max_iterations: 50,
                    timeout_secs: 900, // 15 min default
                    categories: None,
                },
            }),
        }
    }
}

impl TerminalBenchBenchmark {
    pub fn new() -> Self {
        Self::default()
    }
}

impl Benchmark for TerminalBenchBenchmark {
    fn name(&self) -> &str {
        "terminal_bench"
    }

    fn display_name(&self) -> &'static str {
        "TerminalBench 2.1"
    }

    fn category(&self) -> BenchmarkCategory {
        BenchmarkCategory::ToolUse
    }

    fn pre_execute(&self, config: &yaml_serde::Value) -> Result<()> {
        let num_samples = config::extract_usize(config, "num_samples");
        let max_iterations = config::extract_usize(config, "max_iterations").unwrap_or(50);
        let timeout_secs = config::extract_u64(config, "timeout_secs").unwrap_or(900);
        let categories = config::extract_string_vec(config, "categories");

        // Download/verify tasks
        let tasks = download_tasks()?;

        let mut state = self.state.lock().unwrap_or_else(|p| p.into_inner());
        state.tasks = filter_tasks(tasks, &categories, num_samples);
        state.config = TerminalBenchConfig {
            num_samples,
            max_iterations,
            timeout_secs,
            categories,
        };
        state.current_idx = 0;

        println!("  TerminalBench: {} tasks loaded", state.tasks.len());
        Ok(())
    }

    fn execute_one(
        &self,
        model: &Model,
        _config: &yaml_serde::Value,
        tracker: &mut TokenTracker,
    ) -> Result<Option<TaskResult>> {
        let (task, cfg, idx) = {
            let mut state = self.state.lock().unwrap_or_else(|p| p.into_inner());
            if state.current_idx >= state.tasks.len() {
                return Ok(None);
            }
            let idx = state.current_idx;
            let task = state.tasks[idx].clone();
            let cfg = state.config.clone();
            state.current_idx += 1;
            (task, cfg, idx)
        };

        let result = execute_single_task(&task, idx, model, &cfg, tracker)?;
        // Token counts populated by runner.rs from tracker delta
        Ok(Some(result))
    }

    fn to_report_result(&self, b: &BenchmarkResult) -> Result<BenchmarkResult> {
        let raw = &b.raw;
        let pass_rate = raw.get("pass_rate").and_then(|v| v.as_f64()).unwrap_or(0.0);
        let total_tasks = raw.get("total_tasks").and_then(|v| v.as_i64()).unwrap_or(0);
        let passed_tasks = raw
            .get("passed_tasks")
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
            "pass_rate".to_string(),
            Score::float(pass_rate, ScoreUnit::Percent)
                .primary(true)
                .higher_is_better(true)
                .display(format!("{:.1}%", pass_rate * 100.0)),
        );
        scores.insert(
            "tasks_passed".to_string(),
            Score::integer(passed_tasks, ScoreUnit::Count)
                .display(format!("{}/{}", passed_tasks, total_tasks)),
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

        // Category breakdown
        let mut breakdowns = BTreeMap::new();
        if let Some(per_task) = raw.get("per_task").and_then(|v| v.as_array()) {
            let mut category_counts: BTreeMap<String, (i64, i64)> = BTreeMap::new(); // (passed, total)

            for task in per_task {
                let categories: Vec<String> = task
                    .get("categories")
                    .and_then(|v| v.as_array())
                    .map(|arr| {
                        arr.iter()
                            .filter_map(|v| v.as_str().map(|s| s.to_string()))
                            .collect()
                    })
                    .unwrap_or_default();
                let passed = task
                    .get("passed")
                    .and_then(|v| v.as_bool())
                    .unwrap_or(false);

                for cat in &categories {
                    let (p, t) = category_counts.entry(cat.clone()).or_insert((0, 0));
                    *t += 1;
                    if passed {
                        *p += 1;
                    }
                }
            }

            if !category_counts.is_empty() {
                let mut rows = BTreeMap::new();
                for (cat, (passed, total)) in &category_counts {
                    let rate = if *total > 0 {
                        *passed as f64 / *total as f64
                    } else {
                        0.0
                    };
                    rows.insert(
                        cat.clone(),
                        BTreeMap::from([
                            (
                                "pass_rate".to_string(),
                                Score::float(rate, ScoreUnit::Percent)
                                    .display(format!("{:.1}%", rate * 100.0)),
                            ),
                            (
                                "tasks".to_string(),
                                Score::integer(*total, ScoreUnit::Count)
                                    .display(format!("{}/{}", passed, total)),
                            ),
                        ]),
                    );
                }
                breakdowns.insert(
                    "By Category".to_string(),
                    BreakdownTable {
                        title: "Pass Rate by Category".to_string(),
                        rows,
                    },
                );
            }
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
}

/// Download TerminalBench 2.1 tasks from GitHub
fn download_tasks() -> Result<Vec<TerminalBenchTask>> {
    let cache_dir = cache_dir()?;
    let tasks_dir = cache_dir.join("tasks");

    // Check cache first
    if tasks_dir.exists() {
        return load_tasks_from_dir(&tasks_dir);
    }

    // Download from GitHub
    println!("  Downloading TerminalBench 2.1 tasks...");
    std::fs::create_dir_all(&tasks_dir)?;

    // Fetch dataset.toml for task list
    let dataset_url =
        "https://raw.githubusercontent.com/harbor-framework/terminal-bench-2-1/main/tasks/dataset.toml";
    let dataset_bytes =
        crate::download::download_with_retry_bytes(dataset_url, 3, 30, "llm-benchmark-runner")?;
    let dataset_content = String::from_utf8_lossy(&dataset_bytes).to_string();
    let dataset: toml::Value =
        toml::from_str(&dataset_content).context("Failed to parse dataset.toml")?;

    // Get task list from dataset
    let task_names: Vec<String> = dataset
        .get("tasks")
        .and_then(|v| v.as_array())
        .map(|arr| {
            arr.iter()
                .filter_map(|v| v.as_str().map(|s| s.to_string()))
                .collect()
        })
        .unwrap_or_default();

    // Download each task's task.toml and instruction.md
    for task_name in &task_names {
        let task_dir = tasks_dir.join(task_name);
        std::fs::create_dir_all(&task_dir)?;

        // Download task.toml
        let task_url = format!(
            "https://raw.githubusercontent.com/harbor-framework/terminal-bench-2-1/main/tasks/{}/task.toml",
            task_name
        );
        if let Ok(bytes) =
            crate::download::download_with_retry_bytes(&task_url, 2, 30, "llm-benchmark-runner")
        {
            let content = String::from_utf8_lossy(&bytes).to_string();
            fs::write(task_dir.join("task.toml"), &content)?;
        }

        // Download instruction.md
        let instr_url = format!(
            "https://raw.githubusercontent.com/harbor-framework/terminal-bench-2-1/main/tasks/{}/instruction.md",
            task_name
        );
        if let Ok(bytes) =
            crate::download::download_with_retry_bytes(&instr_url, 2, 30, "llm-benchmark-runner")
        {
            let content = String::from_utf8_lossy(&bytes).to_string();
            fs::write(task_dir.join("instruction.md"), content)?;
        }
    }

    load_tasks_from_dir(&tasks_dir)
}

fn load_tasks_from_dir(tasks_dir: &Path) -> Result<Vec<TerminalBenchTask>> {
    let mut tasks = Vec::new();

    for entry in fs::read_dir(tasks_dir)? {
        let entry = entry?;
        let task_toml = entry.path().join("task.toml");
        if task_toml.exists() {
            let content = fs::read_to_string(&task_toml)?;
            if let Ok(task) =
                parse_task_toml(&content, entry.file_name().to_string_lossy().to_string())
            {
                tasks.push(task);
            }
        }
    }

    tasks.sort_by(|a, b| {
        a.metadata
            .task_info
            .as_ref()
            .map(|t| &t.name)
            .unwrap_or(&String::new())
            .cmp(
                b.metadata
                    .task_info
                    .as_ref()
                    .map(|t| &t.name)
                    .unwrap_or(&String::new()),
            )
    });

    Ok(tasks)
}

fn parse_task_toml(content: &str, default_name: String) -> Result<TerminalBenchTask> {
    let mut task: TerminalBenchTask = toml::from_str(content)?;

    // Ensure task name is set
    if task
        .metadata
        .task_info
        .as_ref()
        .map(|t| t.name.is_empty())
        .unwrap_or(true)
    {
        if let Some(ref mut info) = task.metadata.task_info {
            info.name = default_name.clone();
        } else {
            task.metadata.task_info = Some(TaskInfo {
                name: default_name,
                description: String::new(),
                keywords: Vec::new(),
            });
        }
    }

    Ok(task)
}

fn filter_tasks(
    tasks: Vec<TerminalBenchTask>,
    categories: &Option<Vec<String>>,
    num_samples: Option<usize>,
) -> Vec<TerminalBenchTask> {
    let tasks = if let Some(cats) = categories {
        tasks
            .into_iter()
            .filter(|t| {
                let cat = t
                    .metadata
                    .metadata
                    .as_ref()
                    .map(|m| m.category.as_str())
                    .unwrap_or("other");
                cats.iter().any(|c| c.as_str() == cat)
            })
            .collect()
    } else {
        tasks
    };

    match num_samples {
        Some(n) => tasks.into_iter().take(n).collect(),
        None => tasks,
    }
}

fn cache_dir() -> Result<PathBuf> {
    let dir = dirs::cache_dir()
        .ok_or_else(|| anyhow::anyhow!("No cache directory"))?
        .join("llm-benchmark-runner")
        .join("terminal_bench");
    std::fs::create_dir_all(&dir)?;
    Ok(dir)
}

/// Execute a single TerminalBench task
fn execute_single_task(
    task: &TerminalBenchTask,
    idx: usize,
    model: &Model,
    config: &TerminalBenchConfig,
    tracker: &mut TokenTracker,
) -> Result<TaskResult> {
    let task_name = task
        .metadata
        .task_info
        .as_ref()
        .map(|t| t.name.as_str())
        .unwrap_or("unknown");
    let categories = vec![
        task.metadata
            .metadata
            .as_ref()
            .map(|m| m.category.clone())
            .unwrap_or_default(),
        task.metadata
            .metadata
            .as_ref()
            .map(|m| m.difficulty.clone())
            .unwrap_or_default(),
    ];

    // Get instruction
    let cache_dir = cache_dir()?;
    let instruction_path = cache_dir
        .join("tasks")
        .join(task_name)
        .join("instruction.md");
    let instruction = fs::read_to_string(&instruction_path).unwrap_or_default();

    if instruction.is_empty() {
        return Ok(TaskResult::new(
            format!("task-{}", idx),
            false,
            0.0,
            categories,
        ));
    }

    // Get Docker image
    let docker_image = task
        .environment
        .as_ref()
        .map(|e| e.docker_image.clone())
        .unwrap_or_else(|| "ubuntu:22.04".to_string());

    println!("  TerminalBench: {} (image: {})", task_name, docker_image);

    // Start detached container
    // Security is intentionally relaxed: the agent needs to write files, execute
    // arbitrary commands, and manage processes inside the container.
    // Network access is controlled per-task via allow_internet.
    let run_config = DockerRunConfig {
        image: docker_image.clone(),
        command: Vec::new(),
        mounts: Vec::new(),
        workdir: None,
        env: Vec::new(),
        timeout_secs: config.timeout_secs,
        host_repo_path: None,
        network_none: !task
            .environment
            .as_ref()
            .map(|e| e.allow_internet)
            .unwrap_or(false),
        read_only_root: true,
        tmpfs: vec![
            "/tmp:rw,noexec,nosuid,size=64m".to_string(),
            "/workspace:rw,noexec,nosuid,size=256m".to_string(),
            "/tests:rw,nosuid,size=64m".to_string(),
        ],
        cap_drop_all: true,
        no_new_privileges: true,
        pids_limit: Some(128),
        memory: Some(format!(
            "{}m",
            task.environment
                .as_ref()
                .map(|e| e.memory_mb)
                .unwrap_or(4096)
        )),
        name_prefix: "terminal-bench".to_string(),
    };

    let container = match DockerRunner::run_detached(&run_config) {
        Ok(c) => c,
        Err(e) => {
            eprintln!("  Failed to start container for {}: {}", task_name, e);
            return Ok(TaskResult::new(
                format!("task-{}", idx),
                false,
                0.0,
                categories,
            ));
        }
    };

    // Ensure container is always stopped, even on errors
    struct ContainerGuard(String);
    impl Drop for ContainerGuard {
        fn drop(&mut self) {
            let _ = DockerRunner::stop(&self.0);
        }
    }
    let _guard = ContainerGuard(container.clone());

    // System prompt for shell agent
    let system_prompt = r#"You are an expert terminal agent. You have access to a Linux shell.
Execute commands one at a time. Each command you provide will be executed in the container.
You will see the output of each command and can decide what to do next.
When you complete the task, output a line containing only "DONE" to signal completion.
Always provide exactly one shell command per response, wrapped in a tool call to "shell_exec"."#;

    let user_prompt = format!(
        "## Task\n{}\n\n## Instructions\nExecute shell commands to complete the task. Send one command at a time using the shell_exec tool.",
        instruction
    );

    // Define shell_exec tool
    let tools = vec![serde_json::json!({
        "type": "function",
        "function": {
            "name": "shell_exec",
            "description": "Execute a shell command in the terminal",
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {
                        "type": "string",
                        "description": "The shell command to execute"
                    }
                },
                "required": ["command"]
            }
        }
    })];

    // Agentic loop — client manages conversation history internally
    for iteration in 0..config.max_iterations {
        // use_history=true on first call (sends system+user), and on subsequent calls
        // (sends accumulated history + system + a continuation prompt)
        let (text, tool_calls) = if iteration == 0 {
            tracker.chat_completion_with_tools(
                &model.model_name,
                system_prompt,
                &user_prompt,
                tools.clone(),
                None,
                true, // use_history
            )?
        } else {
            // Subsequent turns: send a continuation prompt; history carries prior context
            tracker.chat_completion_with_tools(
                &model.model_name,
                system_prompt,
                "Continue executing commands to complete the task. What is your next command?",
                tools.clone(),
                None,
                true, // use_history
            )?
        };

        // Check for DONE signal
        if text.trim().to_uppercase() == "DONE"
            || (text.lines().any(|l| l.trim().to_uppercase() == "DONE"))
        {
            println!("  TerminalBench: {} completed (DONE signal)", task_name);
            break;
        }

        // Execute tool calls and feed results back into history
        let mut executed_any = false;
        for tool_call in &tool_calls {
            if tool_call.name == "shell_exec" {
                if let Some(cmd) = tool_call.arguments.get("command").and_then(|v| v.as_str()) {
                    if cmd.is_empty() {
                        continue;
                    }
                    executed_any = true;

                    // Execute command in container
                    let output = match DockerRunner::exec(&container, cmd) {
                        Ok(out) => out,
                        Err(e) => format!("Error: {}", e),
                    };

                    // Truncate long outputs and append to client history
                    let truncated = output.chars().take(4000).collect::<String>();
                    tracker.append_tool_result(&tool_call.id, &truncated);
                }
            }
        }

        // If no tool calls were executed, the model may be stuck; break to avoid infinite loop
        if !executed_any && tool_calls.is_empty() {
            println!(
                "  TerminalBench: {} no tool calls, stopping iteration",
                task_name
            );
            break;
        }
    }

    // Run validation (guard ensures container cleanup)
    let passed = validate_task(&container, task_name, &cache_dir)?;

    Ok(TaskResult::new(
        format!("task-{}", idx),
        passed,
        if passed { 1.0 } else { 0.0 },
        categories.clone(),
    )
    .with_metadata(Some(serde_json::json!({
        "task_name": task_name,
        "categories": categories,
        "docker_image": docker_image,
    }))))
}

/// Validate task by running the official test suite.
///
/// Terminal-Bench 2.1 tasks have a `tests/` directory containing `test.sh` and
/// `test_outputs.py`. The `test.sh` script installs dependencies (apt-get, pip, etc.)
/// and runs pytest, then writes `1` (pass) or `0` (fail) to `/logs/verifier/reward.txt`.
fn validate_task(container: &str, task_name: &str, cache_dir: &Path) -> Result<bool> {
    let tests_dir = cache_dir.join("tasks").join(task_name).join("tests");

    // If no test directory in cache, try to download it
    if !tests_dir.exists() {
        let base_url =
            "https://raw.githubusercontent.com/harbor-framework/terminal-bench-2-1/main/tasks";

        // Download test.sh
        let test_sh_url = format!("{}/{}/tests/test.sh", base_url, task_name);
        if let Ok(bytes) =
            crate::download::download_with_retry_bytes(&test_sh_url, 2, 30, "llm-benchmark-runner")
        {
            std::fs::create_dir_all(&tests_dir)?;
            fs::write(tests_dir.join("test.sh"), bytes)?;
        }

        // Download test_outputs.py (common Python test file)
        let test_py_url = format!("{}/{}/tests/test_outputs.py", base_url, task_name);
        if let Ok(bytes) =
            crate::download::download_with_retry_bytes(&test_py_url, 2, 30, "llm-benchmark-runner")
        {
            fs::write(tests_dir.join("test_outputs.py"), bytes)?;
        }
    }

    // Check for test.sh first — this is the standard entrypoint
    if tests_dir.join("test.sh").exists() {
        // Copy test files into container
        for entry in fs::read_dir(&tests_dir)? {
            let entry = entry?;
            let src = entry.path();
            let _ = DockerRunner::cp_to(container, &src, "/tests/");
        }

        // Ensure /logs/verifier directory exists in container
        let _ = DockerRunner::exec(container, "mkdir -p /logs/verifier");

        // Run test.sh which installs deps and writes reward.txt
        let test_output = DockerRunner::exec(
            container,
            "cd /tests && chmod +x test.sh && bash test.sh 2>&1 | tail -50",
        )
        .unwrap_or_default();

        // Check reward.txt for pass/fail
        let reward = DockerRunner::exec(
            container,
            "cat /logs/verifier/reward.txt 2>/dev/null || echo '0'",
        )?;

        if reward.trim() == "1" {
            return Ok(true);
        }

        // If reward.txt wasn't written but test_output looks successful
        if test_output.contains("passed")
            && !test_output.contains("failed")
            && !test_output.contains("ERROR")
        {
            return Ok(true);
        }

        return Ok(false);
    }

    // Fallback: check reward.txt directly (some tasks may have pre-existing verifier)
    let reward = DockerRunner::exec(
        container,
        "cat /logs/verifier/reward.txt 2>/dev/null || echo '0'",
    )?;
    Ok(reward.trim() == "1")
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_task(name: &str, category: &str) -> TerminalBenchTask {
        TerminalBenchTask {
            metadata: TaskMetadata {
                task_info: Some(TaskInfo {
                    name: name.to_string(),
                    description: String::new(),
                    keywords: vec![],
                }),
                metadata: Some(TaskDetails {
                    difficulty: "medium".to_string(),
                    category: category.to_string(),
                    tags: vec![],
                }),
            },
            task: None,
            environment: None,
        }
    }

    #[test]
    fn filter_tasks_by_category() {
        let tasks = vec![
            make_task("task1", "file_management"),
            make_task("task2", "shell_commands"),
            make_task("task3", "file_management"),
        ];
        let filtered = filter_tasks(tasks, &Some(vec!["file_management".to_string()]), None);
        assert_eq!(filtered.len(), 2);
        assert_eq!(
            filtered[0]
                .metadata
                .task_info
                .as_ref()
                .map(|t| t.name.as_str()),
            Some("task1")
        );
    }

    #[test]
    fn filter_tasks_num_samples() {
        let tasks = vec![
            make_task("task1", "other"),
            make_task("task2", "other"),
            make_task("task3", "other"),
        ];
        let filtered = filter_tasks(tasks, &None, Some(2));
        assert_eq!(filtered.len(), 2);
    }

    #[test]
    fn filter_tasks_none_categories_returns_all() {
        let tasks = vec![make_task("task1", "debugging"), make_task("task2", "linux")];
        let filtered = filter_tasks(tasks, &None, None);
        assert_eq!(filtered.len(), 2);
    }

    #[test]
    fn filter_tasks_empty_categories_returns_none() {
        let tasks = vec![make_task("task1", "linux")];
        let filtered = filter_tasks(tasks, &Some(vec![]), None);
        assert_eq!(filtered.len(), 0);
    }

    #[test]
    fn default_difficulty_is_medium() {
        assert_eq!(default_difficulty(), "medium");
    }

    #[test]
    fn default_category_is_other() {
        assert_eq!(default_category(), "other");
    }

    #[test]
    fn default_memory_is_4096() {
        assert_eq!(default_memory(), 4096);
    }

    #[test]
    fn default_storage_is_10240() {
        assert_eq!(default_storage(), 10240);
    }
}
