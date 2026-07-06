use crate::benchmarks;
use crate::client::Client;
use crate::config::{self, DockerConfig, Model};
use crate::utils::format_duration;
use anyhow::Result;
use std::collections::HashMap;
#[cfg(unix)]
use std::os::unix::process::CommandExt;
use std::process::{Child, Command, Stdio};
use std::time::{Duration, Instant};

/// Returns (model_results_json, successful_benchmarks, failed_benchmarks, per_bench_timings)
/// per_bench_timings: HashMap<benchmark_name, Vec<Duration>> for this model's run
// ponytail: suppress type_complexity, tuple return is shortest code
#[allow(clippy::type_complexity)]
pub fn run_model(
    model: &Model,
    benchmarks: &[String],
    benchmark_config: &HashMap<String, serde_yaml::Value>,
    docker_config: &DockerConfig,
    completed_benchmarks: &[String],
) -> Result<(
    serde_json::Value,
    Vec<String>,
    Vec<String>,
    HashMap<String, Vec<Duration>>,
)> {
    println!("\n  Starting model: {}", model.display_name);
    let process = start_model(&model.cmd)?;
    let mut process_guard = ModelProcessGuard::new(process, model.cmd_stop.clone());

    let client = Client::new(&model.proxy)?;
    if !wait_for_health(&client) {
        return Err(anyhow::anyhow!("Proxy did not become healthy"));
    }

    let mut model_results: HashMap<String, serde_json::Value> = HashMap::new();
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
                .unwrap_or(serde_yaml::Value::Null),
            docker_config,
        );

        if wait_for_health(&client) {
            println!("  Proxy healthy before {}.", bench_name);
            match benchmarks::execute_benchmark(bench_name, model, &bench_cfg) {
                Ok(result) => {
                    model_results.insert(bench_name.to_string(), result);
                    new_successful.push(bench_name.to_string());
                }
                Err(e) => {
                    eprintln!("  ERROR: {} - {}", bench_name, e);
                    model_results.insert(
                        bench_name.to_string(),
                        serde_json::json!({"error": e.to_string()}),
                    );
                    new_failed.push(bench_name.to_string());
                }
            }
        } else {
            let message = "proxy not healthy before benchmark execution";
            eprintln!("  ERROR: {} - {}", bench_name, message);
            model_results.insert(
                bench_name.to_string(),
                serde_json::json!({"error": message}),
            );
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

    Ok((
        serde_json::json!(model_results),
        new_successful,
        new_failed,
        per_bench_timings,
    ))
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
        Self {
            cmd_stop,
            process: Some(process),
        }
    }

    pub fn stop(&mut self) {
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

pub fn wait_for_health(client: &Client) -> bool {
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
