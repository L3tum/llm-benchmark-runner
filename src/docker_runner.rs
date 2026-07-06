use anyhow::{Context, Result};
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

#[derive(Debug, Clone)]
pub struct DockerMount {
    pub source: PathBuf,
    pub target: String,
    pub readonly: bool,
    pub map_host_repo_path: bool,
}

impl DockerMount {
    pub fn readonly(source: impl Into<PathBuf>, target: impl Into<String>) -> Self {
        Self {
            source: source.into(),
            target: target.into(),
            readonly: true,
            map_host_repo_path: true,
        }
    }

    pub fn readwrite(source: impl Into<PathBuf>, target: impl Into<String>) -> Self {
        Self {
            source: source.into(),
            target: target.into(),
            readonly: false,
            map_host_repo_path: true,
        }
    }

    pub fn direct_readwrite(source: impl Into<PathBuf>, target: impl Into<String>) -> Self {
        Self {
            source: source.into(),
            target: target.into(),
            readonly: false,
            map_host_repo_path: false,
        }
    }
}

#[derive(Debug, Clone)]
pub struct DockerBuildConfig {
    pub image: String,
    pub dockerfile: PathBuf,
    pub context: PathBuf,
    pub timeout_secs: u64,
    pub host_repo_path: Option<PathBuf>,
}

impl DockerBuildConfig {
    pub fn new(
        image: impl Into<String>,
        dockerfile: impl Into<PathBuf>,
        context: impl Into<PathBuf>,
        timeout_secs: u64,
    ) -> Self {
        Self {
            image: image.into(),
            dockerfile: dockerfile.into(),
            context: context.into(),
            timeout_secs,
            host_repo_path: None,
        }
    }
}

#[derive(Debug, Clone)]
pub struct DockerRunConfig {
    pub image: String,
    pub command: Vec<String>,
    pub mounts: Vec<DockerMount>,
    pub workdir: Option<String>,
    pub env: Vec<(String, String)>,
    pub timeout_secs: u64,
    pub host_repo_path: Option<PathBuf>,
    pub network_none: bool,
    pub read_only_root: bool,
    pub tmpfs: Vec<String>,
    pub cap_drop_all: bool,
    pub no_new_privileges: bool,
    pub pids_limit: Option<u64>,
    pub memory: Option<String>,
    pub name_prefix: String,
}

impl DockerRunConfig {
    pub fn new(image: impl Into<String>, command: Vec<String>, timeout_secs: u64) -> Self {
        Self {
            image: image.into(),
            command,
            mounts: Vec::new(),
            workdir: None,
            env: Vec::new(),
            timeout_secs,
            host_repo_path: None,
            network_none: true,
            read_only_root: true,
            tmpfs: vec!["/tmp:rw,noexec,nosuid,size=64m".to_string()],
            cap_drop_all: true,
            no_new_privileges: true,
            pids_limit: Some(128),
            memory: Some("512m".to_string()),
            name_prefix: "llm-benchmark-runner".to_string(),
        }
    }
}

#[derive(Debug, Clone)]
pub struct DockerRunResult {
    pub timed_out: bool,
    pub exit_code: Option<i32>,
    pub stdout: String,
    pub stderr: String,
}

impl DockerRunResult {
    pub fn success(&self) -> bool {
        !self.timed_out && self.exit_code == Some(0)
    }
}

pub struct DockerRunner;

impl DockerRunner {
    pub fn image_exists(image: &str) -> Result<bool> {
        let output = Command::new("docker")
            .arg("image")
            .arg("inspect")
            .arg(image)
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .status()
            .with_context(|| {
                "failed to inspect docker image; is Docker installed and available on PATH?"
            })?;
        Ok(output.success())
    }

    pub fn build_image(config: &DockerBuildConfig) -> Result<DockerRunResult> {
        let dockerfile = docker_mount_source(&config.dockerfile, config.host_repo_path.as_deref())?;
        let context = docker_mount_source(&config.context, config.host_repo_path.as_deref())?;
        let mut command = Command::new("docker");
        command
            .arg("build")
            .arg("-f")
            .arg(dockerfile)
            .arg("-t")
            .arg(&config.image)
            .arg(context)
            .stdout(Stdio::piped())
            .stderr(Stdio::piped());
        run_command_with_timeout(command, config.timeout_secs)
    }

    pub fn run(config: &DockerRunConfig) -> Result<DockerRunResult> {
        let container_name = docker_container_name(&config.name_prefix);
        let mut command = Command::new("docker");
        command
            .arg("run")
            .arg("--rm")
            .arg("--name")
            .arg(&container_name);

        if config.network_none {
            command.arg("--network").arg("none");
        }
        if config.read_only_root {
            command.arg("--read-only");
        }
        for tmpfs in &config.tmpfs {
            command.arg("--tmpfs").arg(tmpfs);
        }
        if config.cap_drop_all {
            command.arg("--cap-drop=ALL");
        }
        if config.no_new_privileges {
            command.arg("--security-opt=no-new-privileges");
        }
        if let Some(limit) = config.pids_limit {
            command.arg("--pids-limit").arg(limit.to_string());
        }
        if let Some(memory) = &config.memory {
            command.arg("--memory").arg(memory);
        }
        for mount in &config.mounts {
            let source = if mount.map_host_repo_path {
                docker_mount_source(&mount.source, config.host_repo_path.as_deref())?
            } else {
                mount.source.clone()
            };
            let mode = if mount.readonly { "ro" } else { "rw" };
            command
                .arg("-v")
                .arg(format!("{}:{}:{}", source.display(), mount.target, mode));
        }
        if let Some(workdir) = &config.workdir {
            command.arg("-w").arg(workdir);
        }
        for (key, value) in &config.env {
            command.arg("-e").arg(format!("{}={}", key, value));
        }
        command.arg(&config.image);
        for arg in &config.command {
            command.arg(arg);
        }
        command.stdout(Stdio::piped()).stderr(Stdio::piped());

        run_command_with_timeout_and_cleanup(command, config.timeout_secs, || {
            let _ = Command::new("docker")
                .arg("rm")
                .arg("-f")
                .arg(&container_name)
                .output();
        })
    }
}

fn run_command_with_timeout(command: Command, timeout_secs: u64) -> Result<DockerRunResult> {
    run_command_with_timeout_and_cleanup(command, timeout_secs, || {})
}

fn run_command_with_timeout_and_cleanup(
    mut command: Command,
    timeout_secs: u64,
    cleanup: impl FnOnce(),
) -> Result<DockerRunResult> {
    let mut child = command
        .spawn()
        .with_context(|| "failed to start docker; is Docker installed and available on PATH?")?;
    let deadline = Instant::now() + Duration::from_secs(timeout_secs);
    let mut cleanup = Some(cleanup);
    loop {
        if child.try_wait()?.is_some() {
            let output = child.wait_with_output()?;
            return Ok(DockerRunResult {
                timed_out: false,
                exit_code: output.status.code(),
                stdout: String::from_utf8_lossy(&output.stdout).to_string(),
                stderr: String::from_utf8_lossy(&output.stderr).to_string(),
            });
        }
        if Instant::now() >= deadline {
            if let Some(cleanup) = cleanup.take() {
                cleanup();
            }
            let _ = child.kill();
            let output = child.wait_with_output()?;
            return Ok(DockerRunResult {
                timed_out: true,
                exit_code: None,
                stdout: String::from_utf8_lossy(&output.stdout).to_string(),
                stderr: String::from_utf8_lossy(&output.stderr).to_string(),
            });
        }
        std::thread::sleep(Duration::from_millis(100));
    }
}

fn docker_container_name(prefix: &str) -> String {
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_nanos();
    format!("{}-{}-{}", sanitize_name(prefix), std::process::id(), nanos)
}

fn sanitize_name(value: &str) -> String {
    value
        .chars()
        .map(|c| {
            if c.is_ascii_alphanumeric() || c == '-' || c == '_' || c == '.' {
                c
            } else {
                '-'
            }
        })
        .collect()
}

pub fn docker_mount_source(path: &Path, host_repo_path: Option<&Path>) -> Result<PathBuf> {
    let canonical = path.canonicalize()?;
    if let Some(host_repo_path) = host_repo_path {
        let repo_root = std::env::current_dir()?.canonicalize()?;
        let relative = canonical.strip_prefix(&repo_root).with_context(|| {
            format!(
                "mount source {} is not under repo root {}; cannot apply host_repo_path",
                canonical.display(),
                repo_root.display()
            )
        })?;
        Ok(host_repo_path.join(relative))
    } else {
        Ok(canonical)
    }
}
