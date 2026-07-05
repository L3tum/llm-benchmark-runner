# Rewrite `llm-benchmark-runner` in Rust

## Context

The current project is a Python benchmark runner for LLM models. It:
- Reads a YAML config (`models_config.yaml`) defining models, their start/stop commands, and proxy URLs.
- Starts models via `llama-server` (shell commands), waits for proxy readiness.
- Runs benchmarks (MMLU-Pro accuracy, KLD pairwise) against the model's OpenAI-compatible API.
- Saves intermediate results atomically to a JSON file.
- Generates Markdown and HTML reports.

The goal is to rewrite the project in Rust, following the build/release/CI patterns of the LTEngine project (static musl binary, Docker, CI workflows).

## Approach

### Project Structure

```
llm-benchmark-runner/
├── Cargo.toml
├── src/
│   ├── main.rs            # CLI entry (subcommands: run, report)
│   ├── config.rs          # YAML config parsing, Model/Benchmark definitions
│   ├── runner.rs          # Model lifecycle (start, wait, stop, sigterm/sigkill)
│   ├── benchmarks/
│   │   ├── mod.rs         # Benchmark trait and registry
│   │   ├── mmlu_pro.rs    # MMLU-Pro evaluation
│   │   └── kld.rs         # KLD computation
│   ├── client.rs          # OpenAI-compatible HTTP client (reqwest)
│   └── report.rs          # Markdown + HTML report generation (askama templates)
├── templates/             # Askama templates for HTML
│   └── report.html
├── models_config.yaml     # example config
├── Cargo.lock
└── ...
```

### Key Rust Crates

- **CLI**: `clap`
- **Async runtime**: `tokio`
- **HTTP**: `reqwest`
- **YAML**: `serde`, `serde_yaml`
- **JSON**: `serde_json`
- **Templates**: `askama` (with `askama_derive`)
- **Progress bars**: `console::ProgressBar` (or `indicatif`)
- **Process management**: `tokio::process` or `nix`/`libc` for process groups (setsid, killpg)
- **KLD / math**: `ndarray` or plain iterators, `log` from standard `f64`
- **Datasets (MMLU-Pro)**: `datasets-rs` or direct HTTP fetch from HuggingFace (simpler, no binary dep)
- **Error handling**: `anyhow`

### MMLU-Pro Data Handling

The Python version uses `datasets` library to load `TIGER-Lab/MMLU-Pro` from HuggingFace. In Rust, we can either:
- Use `datasets-rs` (binds to Python datasets library — heavy).
- **Preferred**: Use `reqwest` to download the parquet files directly from HuggingFace, then parse with `parquet2` or convert to JSON. Simpler: just download the dataset as a JSON archive (HuggingFace allows downloading as `.json`). The dataset is ~40MB JSON. This avoids binary dependencies.

### Benchmark Trait

```rust
pub trait Benchmark: Send + Sync {
    fn name(&self) -> &str;
    fn pre_execute(&self, config: &serde_json::Value) -> Result<()>;
    fn execute(&self, model: &Model, config: &serde_json::Value) -> Result<serde_json::Value>;
    fn post_execute(&self, model_results: &HashMap<String, serde_json::Value>) -> Result<serde_json::Value>;
}
```

The registry is a static HashMap mapping names to boxed `Benchmark` instances.

### Process Management

Use `tokio::process::Command` with `kill_on_drop(true)` for safety. For process groups, use `nix::unistd::setpgid` or `libc::setpgid` in a `pre_exec` callback. On Linux, we can use `libc::setsid` in `pre_exec`. On non-Linux (macOS), fall back to `std::process::Command` with `start_new_session` if available.

### Dockerfile (similar to LTEngine)

Static binary build using `rust:1.96`, add `x86_64-unknown-linux-musl` target, `musl-tools`. Copy source, build with `cargo build --release --target x86_64-unknown-linux-musl`, copy binary to scratch image.

### Makefile (similar to LTEngine)

Targets: `check`, `fmt`, `fmt-check`, `clippy`, `test`, `miri` (with nightly), `deny` (cargo-deny), `all`, `build-release`, `docs`.

### GitHub Workflows

- **build.yml**: lint (fmt, clippy), build-and-test (with system deps like `libclang-dev` for any native bindings), miri (UB detection), deny (licenses, advisories, bans).
- **docker.yml**: Build and push Docker image on release (same pattern as LTEngine).

## Files to Modify

- Create new Rust source tree as above.
- Delete Python files (`benchmark_runner.py`, `report_generator.py`, `manage_llama_swap_models.py`, `benchmarks/`), `requirements.txt`, `venv`.
- Keep `models_config.yaml` as example config (update format if needed).
- Create `Dockerfile`, `Makefile`, `.github/workflows/build.yml`, `.github/workflows/docker.yml`.
- Create `.cargo/config.toml` if needed for deny.
- Create `deny.toml` for cargo-deny config.

## Steps

- [ ] Set up Rust project structure: `cargo init --bin` or `cargo new`.
- [ ] Add dependencies in `Cargo.toml`: clap, tokio, reqwest, serde, serde_yaml, serde_json, askama, console, libc/nix, parquet2 (optional), thiserror, anyhow.
- [ ] Implement config parsing (YAML → structs).
- [ ] Implement OpenAI-compatible client for chat completions and logprobs.
- [ ] Implement benchmark trait and MMLU-Pro benchmark (download dataset, few-shot, evaluate, extract answers).
- [ ] Implement KLD benchmark (collect logits, pairwise KLD).
- [ ] Implement benchmark registry and runner (model lifecycle, result saving, resuming).
- [ ] Implement HTML report generation with askama template.
- [ ] Implement CLI: `run` subcommand with --config and --no-resume flags.
- [ ] Write tests for config parsing, KLD computation, benchmark execute (mock client).
- [ ] Create Dockerfile (static musl build).
- [ ] Create Makefile with lint, test, deny, miri targets.
- [ ] Create GitHub workflow `build.yml` (lint, build, test, miri, deny).
- [ ] Create GitHub workflow `docker.yml` (build and push on release).
- [ ] Create `deny.toml` for license/advisory checking.
- [ ] Verify: build release, run on a test config with a sample model.

## Reuse

- LTEngine's Dockerfile pattern: multi-stage build, musl static binary.
- LTEngine's Makefile targets: check, fmt, clippy, test, miri, deny, docs.
- LTEngine's build.yml: lint, build-and-test, miri, deny jobs.
- LTEngine's docker.yml: build/push Docker on release.

## Verification

- `make all` should pass (check, fmt, clippy, test, deny).
- `cargo build --release` produces a static binary.
- `docker build .` produces a minimal image.
- Run with a real config: start a local llama-server instance, run `cargo run run --config models_config.yaml`, verify results and reports are generated.
- MIRI passes (no UB).
