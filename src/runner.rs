use crate::benchmarks;
use crate::client::Client;
use crate::config::{self, DockerConfig, Model};
use crate::shared::{BenchmarkResult, Diagnostic, TaskResult};
use crate::token_tracker::TokenTracker;
use crate::utils::format_duration;
use anyhow::Result;
use std::collections::HashMap;
#[cfg(unix)]
use std::os::unix::process::CommandExt;
use std::process::{Child, Command, Stdio};
use std::sync::LazyLock;
use std::sync::Mutex;
use std::time::{Duration, Instant};

/// Global tracker for the currently running model process PID.
/// Used by the ctrl-c handler to stop the model gracefully.
pub static CURRENT_MODEL_PID: LazyLock<Mutex<Option<u64>>> = LazyLock::new(|| Mutex::new(None));

/// Runs a model through the given benchmarks and returns (model_results, successful, failed, timings).
/// model_results is a HashMap from benchmark name to the in-memory BenchmarkResult.
// ponytail: suppress type_complexity, tuple return is shortest code
#[allow(clippy::type_complexity)]
pub fn run_model(
    model: &Model,
    benchmarks: &[String],
    benchmark_config: &HashMap<String, yaml_serde::Value>,
    docker_config: &DockerConfig,
    completed_benchmarks: &[String],
) -> Result<(
    HashMap<String, BenchmarkResult>,
    Vec<String>,
    Vec<String>,
    HashMap<String, Vec<Duration>>,
)> {
    println!("\n  Starting model: {}", model.display_name);
    let process = start_model(&model.cmd)?;
    let mut process_guard = ModelProcessGuard::new(process, model.cmd_stop.clone());

    let client = Client::new(&model.proxy)?;
    if let Some(ms) = model.rate_limit_ms {
        client.set_rate_limit(ms);
    }
    if !wait_for_health(&client) {
        return Err(anyhow::anyhow!("Proxy did not become healthy"));
    }

    let mut model_results: HashMap<String, BenchmarkResult> = HashMap::new();
    let mut new_successful = completed_benchmarks.to_vec();
    let mut new_failed = Vec::new();
    let mut per_bench_timings: HashMap<String, Vec<Duration>> = HashMap::new();

    let total_benchmarks = benchmarks.len();
    let mut completed_count = 0;

    for (idx, bench_name) in benchmarks.iter().enumerate() {
        let bench_start = Instant::now();

        let bench_cfg = config::attach_docker_config(
            benchmark_config
                .get(bench_name)
                .cloned()
                .unwrap_or(yaml_serde::Value::Null),
            docker_config,
        );

        if wait_for_health(&client) {
            println!("  Proxy healthy before {}.", bench_name);
            // Try execute_one first; fall back to execute if not supported
            match run_benchmark(bench_name, model, &bench_cfg) {
                Ok(result) => {
                    model_results.insert(bench_name.to_string(), result);
                    new_successful.push(bench_name.to_string());
                }
                Err(e) => {
                    eprintln!("  ERROR: {} - {}", bench_name, e);
                    let error_result = BenchmarkResult {
                        scores: std::collections::BTreeMap::new(),
                        breakdowns: std::collections::BTreeMap::new(),
                        error_classification: std::collections::BTreeMap::new(),
                        artifacts: vec![],
                        diagnostics: vec![Diagnostic {
                            level: "error".to_string(),
                            message: e.to_string(),
                        }],
                        raw: serde_json::json!({"error": e.to_string()}),
                    };
                    model_results.insert(bench_name.to_string(), error_result);
                    new_failed.push(bench_name.to_string());
                }
            }
        } else {
            let message = "proxy not healthy before benchmark execution";
            eprintln!("  ERROR: {} - {}", bench_name, message);
            let error_result = BenchmarkResult {
                scores: std::collections::BTreeMap::new(),
                breakdowns: std::collections::BTreeMap::new(),
                error_classification: std::collections::BTreeMap::new(),
                artifacts: vec![],
                diagnostics: vec![Diagnostic {
                    level: "error".to_string(),
                    message: message.to_string(),
                }],
                raw: serde_json::json!({"error": message}),
            };
            model_results.insert(bench_name.to_string(), error_result);
            new_failed.push(bench_name.to_string());
        }
        let bench_duration = bench_start.elapsed();

        // Record timing
        per_bench_timings
            .entry(bench_name.clone())
            .or_default()
            .push(bench_duration);
        completed_count += 1;

        // Local ETA based on this model's own timings so far
        let remaining_benchmarks = total_benchmarks - completed_count;
        if remaining_benchmarks > 0 {
            let avg = per_bench_timings
                .values()
                .flat_map(|v| v.iter())
                .copied()
                .collect::<Vec<_>>();
            let eta_str = if avg.is_empty() {
                "–".to_string()
            } else {
                let total_so_far: Duration = avg.iter().cloned().sum();
                let mean = total_so_far.div_f64(avg.len() as f64);
                let eta = mean.mul_f64(remaining_benchmarks as f64);
                format_duration(eta)
            };
            let runtime = format_duration(bench_duration);
            println!(
                "  [benchmark {}/{}] {} runtime: {}, ETA: {}",
                idx + 1,
                total_benchmarks,
                bench_name,
                runtime,
                eta_str
            );
        }
    }

    println!("  Stopping model: {}", model.display_name);
    process_guard.stop();

    Ok((model_results, new_successful, new_failed, per_bench_timings))
}
#[cfg(unix)]
pub fn start_model(cmd: &str) -> Result<Child> {
    let process = unsafe {
        Command::new("/bin/bash")
            .arg("-c")
            .arg(cmd)
            .stdin(Stdio::null())
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .pre_exec(|| {
                // SAFETY: `setsid()` is callable in the child just after fork(),
                // before any threads are spawned in the child, so it cannot
                // interfere with another thread's process group membership.
                // It returns -1 on error; we ignore the result to keep the
                // child running even if it fails to create a new session.
                libc::setsid();
                Ok(())
            })
            .spawn()?
    };
    Ok(process)
}
#[cfg(not(unix))]
pub fn start_model(cmd: &str) -> Result<Child> {
    let process = Command::new("/bin/bash")
        .arg("-c")
        .arg(cmd)
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .spawn()?;
    Ok(process)
}
pub struct ModelProcessGuard {
    cmd_stop: Option<String>,
    process: Option<Child>,
}

impl ModelProcessGuard {
    pub fn new(process: Child, cmd_stop: Option<String>) -> Self {
        let pid = process.id() as u64;
        *CURRENT_MODEL_PID
            .lock()
            .expect(crate::shared::MUTEX_PANIC_MSG) = Some(pid);
        Self {
            cmd_stop,
            process: Some(process),
        }
    }

    pub fn stop(&mut self) {
        // Clear the global PID first to prevent double-kills from ctrl-c handler
        *CURRENT_MODEL_PID
            .lock()
            .expect(crate::shared::MUTEX_PANIC_MSG) = None;
        if let Some(process) = self.process.take() {
            stop_model(&self.cmd_stop, process);
        }
    }
}

impl Drop for ModelProcessGuard {
    fn drop(&mut self) {
        self.stop();
    }
}

/// Runs a single benchmark via `execute_one`.
///
/// Creates a `TokenTracker` and passes it to each `execute_one` call. After each task,
/// takes a snapshot of the tracker to compute the per-task token delta and annotates
/// the `TaskResult` with those counts.
fn run_benchmark(
    bench_name: &str,
    model: &Model,
    bench_cfg: &yaml_serde::Value,
) -> Result<BenchmarkResult> {
    // Create tracker for this benchmark run
    let client = Client::new_with_model_params(&model.proxy, model.set_params.as_ref())?;
    if let Some(ms) = model.rate_limit_ms {
        client.set_rate_limit(ms);
    }
    let mut tracker = TokenTracker::new(client);

    let mut task_results: Vec<TaskResult> = Vec::new();

    loop {
        // Snapshot before the task
        let (prev_output, prev_thinking) = tracker.snapshot();
        let (prev_tc_total, prev_tc_valid, prev_tc_invalid) = tracker.tool_call_snapshot();

        match benchmarks::execute_benchmark_one(bench_name, model, bench_cfg, &mut tracker) {
            Ok(Some(mut task_result)) => {
                // Annotate with per-task token delta from tracker
                let (curr_output, curr_thinking) = tracker.snapshot();
                task_result.output_tokens = curr_output.saturating_sub(prev_output);
                task_result.thinking_tokens = curr_thinking.saturating_sub(prev_thinking);

                // Annotate with per-task tool call delta from tracker
                let (curr_tc_total, curr_tc_valid, curr_tc_invalid) = tracker.tool_call_snapshot();
                task_result.tool_calls_total = curr_tc_total.saturating_sub(prev_tc_total);
                task_result.tool_calls_valid = curr_tc_valid.saturating_sub(prev_tc_valid);
                task_result.tool_calls_invalid = curr_tc_invalid.saturating_sub(prev_tc_invalid);

                task_results.push(task_result);
            }
            Ok(None) => {
                // All tasks done
                break;
            }
            Err(e) => {
                return Err(e);
            }
        }
    }

    // Check if benchmark supports batch evaluation
    if let Ok(bench) = benchmarks::get_benchmark(bench_name) {
        match bench.batch_evaluate(&task_results, bench_cfg) {
            Ok(Some(new_results)) => task_results = new_results,
            Ok(None) => { /* batch evaluator chose not to modify results */ }
            Err(e) => {
                return Err(anyhow::anyhow!(
                    "batch evaluation failed for {}: {}",
                    bench_name,
                    e
                ));
            }
        }
    }

    Ok(BenchmarkResult::from_task_results(task_results))
}

pub fn wait_for_health(client: &Client) -> bool {
    // Fast path: a health check succeeded recently, skip redundant roundtrips.
    // Benchmarks within a model share the same Client, so this avoids 80+
    // repeated HTTP health checks across a run.
    if client.recently_healthy(Duration::from_secs(30)) {
        return true;
    }
    const STABLE_HEALTH_CHECKS: usize = 2;
    let timeout = Duration::from_secs(120);
    let poll = Duration::from_secs(2);
    let deadline = Instant::now() + timeout;
    let mut consecutive_successes = 0;

    while Instant::now() < deadline {
        match client.check_health() {
            Ok(_) => {
                consecutive_successes += 1;
                if consecutive_successes >= STABLE_HEALTH_CHECKS {
                    client.mark_healthy();
                    return true;
                }
                std::thread::sleep(poll);
            }
            Err(_) => {
                consecutive_successes = 0;
                std::thread::sleep(poll);
            }
        }
    }
    false
}
pub fn stop_model(cmd_stop: &Option<String>, mut process: Child) {
    let mut stopped = false;
    if let Some(cmd) = cmd_stop {
        if let Ok(output) = Command::new("/bin/bash").arg("-c").arg(cmd).output() {
            if output.status.success() {
                stopped = true;
                std::thread::sleep(Duration::from_secs(1));
            }
        }
    }
    if !stopped {
        #[cfg(unix)]
        {
            let pid = process.id();
            let _ = Command::new("kill")
                .arg("-TERM")
                .arg(format!("-{}", pid))
                .output();
        }
        std::thread::sleep(Duration::from_secs(2));
    }
    // Force kill if still alive
    let pid = process.id();
    let _ = Command::new("kill")
        .arg("-KILL")
        .arg(format!("-{}", pid))
        .output();
    let _ = process.wait();
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::process::Command;

    #[test]
    fn wait_for_health_returns_true_for_healthy_proxy() {
        let mut server = mockito::Server::new();
        let mock = server
            .mock("GET", "/models")
            .with_status(200)
            .with_header("content-type", "application/json")
            .with_body(r#"{"data":[{"id":"test-model"}]}"#)
            .expect(2)
            .create();
        let client = Client::new(&server.url()).unwrap();
        assert!(wait_for_health(&client));
        mock.assert();
    }

    #[test]
    fn model_process_guard_sets_and_clears_pid() {
        let child = Command::new("sh").arg("-c").arg("exit 0").spawn().unwrap();
        let pid = child.id() as u64;
        let mut guard = ModelProcessGuard::new(child, None);
        assert_eq!(
            *CURRENT_MODEL_PID
                .lock()
                .expect(crate::shared::MUTEX_PANIC_MSG),
            Some(pid)
        );
        guard.stop();
        assert_eq!(
            *CURRENT_MODEL_PID
                .lock()
                .expect(crate::shared::MUTEX_PANIC_MSG),
            None
        );
    }
}
