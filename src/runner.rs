use crate::benchmarks;
use crate::client::Client;
use crate::config::Model;
use anyhow::Result;
use std::collections::HashMap;
#[cfg(unix)]
use std::os::unix::process::CommandExt;
use std::process::{Child, Command, Stdio};
use std::time::{Duration, Instant};
pub fn run_model(
    model: &Model,
    benchmarks: &[String],
    benchmark_config: &HashMap<String, serde_yaml::Value>,
    completed_benchmarks: &[String],
) -> Result<(serde_json::Value, Vec<String>, Vec<String>)> {
    println!("\n  Starting model: {}", model.display_name);
    let process = start_model(&model.cmd)?;

    let client = Client::new(&model.proxy)?;
    if !wait_for_health(&client) {
        stop_model(&model.cmd_stop, process);
        return Err(anyhow::anyhow!("Proxy did not become healthy"));
    }

    let mut model_results: HashMap<String, serde_json::Value> = HashMap::new();
    let mut new_successful = completed_benchmarks.to_vec();
    let mut new_failed = Vec::new();

    for bench_name in benchmarks {
        let bench_cfg = benchmark_config
            .get(bench_name)
            .cloned()
            .unwrap_or(serde_yaml::Value::Null);
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
    }

    println!("  Stopping model: {}", model.display_name);
    stop_model(&model.cmd_stop, process);

    Ok((serde_json::json!(model_results), new_successful, new_failed))
}
#[cfg(unix)]
fn start_model(cmd: &str) -> Result<Child> {
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
fn start_model(cmd: &str) -> Result<Child> {
    let process = Command::new("/bin/bash")
        .arg("-c")
        .arg(cmd)
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .spawn()?;
    Ok(process)
}
fn wait_for_health(client: &Client) -> bool {
    let timeout = Duration::from_secs(120);
    let poll = Duration::from_secs(2);
    let deadline = Instant::now() + timeout;
    while Instant::now() < deadline {
        match client.check_health() {
            Ok(_) => return true,
            Err(_) => std::thread::sleep(poll),
        }
    }
    false
}
fn stop_model(cmd_stop: &Option<String>, mut process: Child) {
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
