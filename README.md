# Model Benchmark Suite

A Rust benchmark suite for evaluating LLM models with **direct model execution**: launch each model, benchmark against its local API, then stop it. Supports **MMLU-Pro** (auto-downloaded), **GPQA Diamond**, **AIME 2025**, **MATH-500**, **Coding Eval**, **SWE-Bench**, and **KLD divergence** between models.

## Quick Start

**GPQA Diamond** (gated dataset) requires a HuggingFace token:

```bash
export HF_TOKEN="hf_..."
```

Then configure and run:
```bash
cargo run -- run --config models_config.yaml
```

## Features

- **Direct execution**: each model starts with a custom `cmd`, benchmarks run against its local OpenAI-compatible proxy, then `cmd_stop` tears it down.
- **MMLU-Pro**: benchmark with up to 10 options per question, CoT few-shot prompting, per-subject accuracy.
- **KLD**: pairwise Kullback-Leibler divergence between model distributions on shared prompts.
- **GPQA Diamond**: 198 graduate-level science questions (biology, chemistry, physics) with zero-shot CoT.
- **AIME 2025**: 30 competition-level math problems with integer answer extraction.
- **MATH-500**: 500 math problems across 7 subjects with per-subject accuracy.
- **Coding Eval**: Docker-backed function-completion evaluation with explicit HumanEval, HumanEval+, and MBPP+ benchmark names; EvalPlus-style evaluation keeps the oracle separated from generated code.
- **SWE-Bench**: Docker-backed repository patch benchmarks for SWE-Bench Basic, Verified, and Pro-style datasets.
- **Resumable**: results saved after each model; rerunning skips completed models.
- **Extensible benchmarks**: add a new benchmark by creating a module in `src/benchmarks/` and registering it in `src/benchmarks/mod.rs`.
- **CLI config override**: `--config path/to/config.yaml` for all commands.

## Prerequisites

- **Rust 1.83+** (stable)
- A GGUF model and a runner (llama-server, ollama, etc.) that exposes an OpenAI-compatible API
- **Docker** if you enable code/repository benchmarks (`humaneval_plus`, `mbpp_plus`, `swebench_verified`, etc.). Mounting `/var/run/docker.sock` into the runner container gives it host-level Docker control; only do this in trusted environments.

## Installation

```bash
cargo install --path .
```

## Configuration

Edit `models_config.yaml`:

```yaml
models:
  - display_name: "MyModel Q4"
    model: "model"  # required: model name to send to the proxy API
    cmd: "llama-server -m /path/to/model.Q4.gguf --port 28287"
    proxy: "http://localhost:28287/v1"
    cmd_stop: "pkill -9 llama-server"   # optional, defaults to SIGTERM

  - display_name: "MyModel Q5"
    model: "model"
    cmd: "llama-server -m /path/to/model.Q5.gguf --port 28288"
    proxy: "http://localhost:28288/v1"

benchmarks:
  - mmlu_pro
  - kld
  - gpqa
  - aime
  - math500
  # - humaneval_plus
  # - mbpp_plus
  # - swebench_verified

# Shared Docker runtime for code/repository benchmarks.
docker:
  enabled: true
  default_timeout_secs: 8
  images:
    python: python:3.12
    swebench_harness: llm-benchmark-runner/swebench-harness:latest
  build_images: true
  max_workers: 1
  mount_docker_socket: true       # needed by official SWE-Bench harness containers
  docker_socket_path: /var/run/docker.sock
  # host_repo_path: /host/path/to/llm-benchmark-runner

benchmark:
  mmlu_pro:
    num_samples: 100
    # subjects: null  # filter by subjects (comma-separated), null = all
  kld:
    num_prompts: 50
    prompt_source: mmlu
    # custom_prompts_path: null
  gpqa:
    # num_samples: 10   # use all 198 questions if omitted
    # subjects: "physics"   # filter by category (comma-separated): biology, chemistry, physics
    # Note: requires HF_TOKEN environment variable for gated dataset
  aime:
    # num_samples: 30   # use all 30 if omitted
  math500:
    # num_samples: 50   # use all 500 if omitted
    # subjects: "algebra"   # filter by subject (comma-separated), null = all subjects
  humaneval_plus:
    num_samples: 10
    timeout_secs: 8
    enable_pass2: false
    enable_pass3: false
  mbpp_plus:
    num_samples: 10
  swebench_verified:
    num_samples: 1
    split: test
    timeout_secs: 1800
  swebench_pro:
    num_samples: 1
    split: test
    timeout_secs: 1800
    token_env: HF_TOKEN
    # dataset_id: SWE-bench/SWE-bench_Pro
```

Each model is:
1. Started with `cmd` (shell command)
2. Benchmark tests run against `proxy` using the `model` name in API calls
3. Stopped with `cmd_stop` (optional, defaults to SIGTERM)
4. Results saved after each model for resumability

The `benchmarks` list controls which benchmarks to run (default: all registered). The `benchmark` section holds per-benchmark configuration.

## Running Benchmarks

```bash
cargo run -- run --config models_config.yaml
```

The script will:
1. Start model 1, run MMLU-Pro and collect KLD logits, stop it, save results
2. Start model 2, run MMLU-Pro and collect KLD logits, stop it, save results
3. Compute pairwise KLD and generate reports

Results are saved to `benchmark_results/`:
- `results.json` — raw benchmark data
- `benchmark_report.md` — Markdown report
- `benchmark_report.html` — styled HTML report

**Resuming**: kill the process, rerun — it will skip models already completed. Use `--no-resume` to force a full re-run.

## Testing Models (Without Running Benchmarks)

Before running full benchmarks, validate each model's configuration (start, health check, prompt, stop):

```bash
cargo run -- test-models --config models_config.yaml
```

This command:
1. Starts each model with `cmd`
2. Waits for the proxy to become healthy
3. Sends a simple test prompt: "Say hello in one word."
4. Stops the model with `cmd_stop` (or SIGTERM/SIGKILL)
5. Prints a summary table (PASS/FAIL) for each model

Useful for CI/CD pipelines or quickly verifying config changes without spending time on full benchmarks.

## Generating Reports

Reports are automatically generated after benchmarking. To generate reports from existing results without rerunning benchmarks:

```bash
cargo run -- report
```

Additional options: `--results` (path to results JSON, default: `benchmark_results/results.json`) and `--output` (output directory, default: `benchmark_results`).

## Adding New Benchmarks

1. Create `src/benchmarks/my_bench.rs` with a struct that implements the `Benchmark` trait (see `mmlu_pro.rs` and `kld.rs` for examples):
   ```rust
   pub struct MyBenchmark;

   impl Benchmark for MyBenchmark {
       fn name(&self) -> &str { "my_bench" }
       
       fn execute(&self, model: &Model, config: &serde_yaml::Value) -> Result<serde_json::Value> {
           // Run your benchmark logic
           Ok(serde_json::json!({"my_metric": 0.8}))
       }
   }
   ```
2. Register the benchmark in `src/benchmarks/mod.rs`:
   ```rust
   pub mod my_bench;
   // In registry():
   map.insert("my_bench".to_string(), Box::new(my_bench::MyBenchmark));
   ```
3. Add `"my_bench"` to the `benchmarks` list in your config, or leave it empty to include all registered benchmarks.

## MMLU-Pro Details

The MMLU-Pro benchmark:
- Auto-downloads from HuggingFace (`TIGER-Lab/MMLU-Pro`)
- Handles up to 10 options (A–J), filtering `N/A` options
- Uses few-shot CoT examples from the validation set
- Answer extraction: regex patterns matching `The answer is (X)` or similar
- Reports per-subject accuracy

## KLD Divergence

After each model is benchmarked, its logprobs are collected for the KLD prompts (defaults to MMLU-Pro questions). At the end, pairwise KL divergence is computed between all model pairs. Lower KLD means more similar output distributions.

## 2026 Modern Benchmarks

The suite includes three cutting-edge 2026 LLM evaluation benchmarks:

### GPQA Diamond

Graduate-level science multiple-choice benchmark (198 questions) covering **biology**, **chemistry**, and **physics**. Uses zero-shot chain-of-thought prompting with A-D answer extraction.

**⚠️ HF_TOKEN required**: The GPQA Diamond dataset is gated on HuggingFace. You must set the `HF_TOKEN` environment variable before running this benchmark:

```bash
export HF_TOKEN="hf_..."
```

Configuration:
```yaml
gpqa:
  num_samples: 10       # use all 198 questions if omitted
  subjects: "physics"   # filter by category (comma-separated): biology, chemistry, physics
```

### AIME (American Invitational Mathematics Examination)

Competition-level math problems with integer answer extraction from `\boxed{}` notation. Answers are 3-digit integers (000–999). Supports **AIME 2025** and **AIME 2026** datasets via the `year` config option.

Datasets from **MathArena** (`MathArena/aime_2025` or `MathArena/aime_2026`), downloaded via the HuggingFace datasets viewer API.

Configuration:
```yaml
aime:
  num_samples: 30       # use all 30 problems if omitted
  year: "2025"          # "2025" or "2026", default 2025
```

### MATH-500

500 competition-level math problems across 7 subjects (algebra, geometry, number theory, precalculus, etc.). Zero-shot chain-of-thought with integer answer extraction.

Configuration:
```yaml
math500:
  num_samples: 50      # use all 500 if omitted
  subjects: "algebra"   # filter by subject (comma-separated), null = all subjects
```

### Code and Repository Benchmarks

Docker-backed code benchmarks are selected by benchmark name:

- `humaneval_plus` — EvalPlus HumanEval+ with oracle-separated evaluation
- `mbpp_plus` — EvalPlus MBPP+ with oracle-separated evaluation
- `humaneval` — original OpenAI HumanEval; note that original HumanEval contains public tests
- `swebench` — SWE-Bench Basic/full dataset
- `swebench_verified` — SWE-Bench Verified
- `swebench_pro` — SWE-Bench Pro-style gated/private dataset ID, configurable per installation

Generated code is written under `benchmark_results/coding_eval_runs/...`; SWE-Bench predictions are written under `benchmark_results/swe_bench_runs/...`. Function-completion containers run with no network and read-only mounts. SWE-Bench harness containers may need network/build access to prepare repositories and environments.

The project Docker image includes the Docker CLI. If running this runner inside Docker, mounting `/var/run/docker.sock` allows it to start sibling containers but also grants powerful access to the host Docker daemon. Use only in trusted environments. If using Docker socket passthrough, set top-level `docker.host_repo_path` to the host-visible repository path so bind mounts resolve correctly.

Configuration:
```yaml
benchmarks:
  - humaneval_plus
  - mbpp_plus
  - swebench_verified

docker:
  enabled: true
  default_timeout_secs: 8
  images:
    python: python:3.12
    swebench_harness: llm-benchmark-runner/swebench-harness:latest
  build_images: true
  max_workers: 1
  mount_docker_socket: true
  docker_socket_path: /var/run/docker.sock
  # host_repo_path: /home/me/llm-benchmark-runner

benchmark:
  humaneval_plus:
    num_samples: 10
    timeout_secs: 8
    enable_pass2: true
    enable_pass3: false
  mbpp_plus:
    num_samples: 10
  swebench_verified:
    num_samples: 1
    split: test
    timeout_secs: 1800
  swebench_pro:
    num_samples: 1
    split: test
    timeout_secs: 1800
    token_env: HF_TOKEN
    # dataset_id: SWE-bench/SWE-bench_Pro
```

`enable_pass2`/`enable_pass3` use iterative repair, not independent sampling: the failed solution plus Docker stdout/stderr/error summary is fed back to the model. If an earlier attempt passes, later enabled attempts are marked passed and skipped without another model call.

SWE-Bench datasets are downloaded and cached automatically from HuggingFace dataset row APIs. Verified and Basic are public; Pro may require `HF_TOKEN` and/or a `dataset_id` override depending on how access is provisioned. SWE-Bench runs can be slow and expensive because each instance may build a repository-specific environment.

## Example Full Configuration

```yaml
models:
  - display_name: "MyModel Q4"
    model: "model"
    cmd: "llama-server -m /path/to/model.Q4.gguf --port 28287"
    proxy: "http://localhost:28287/v1"

benchmarks:
  - mmlu_pro
  - kld
  - gpqa
  - aime
  - math500

benchmark:
  mmlu_pro:
    num_samples: 100
  kld:
    num_prompts: 50
    prompt_source: mmlu
  gpqa:
    num_samples: 10
    subjects: "physics"
  aime:
    num_samples: 10
  math500:
    num_samples: 20
    subjects: "algebra"
```

Remember to set `HF_TOKEN` before running if you include GPQA:
```bash
export HF_TOKEN="hf_..."
cargo run -- run --config models_config.yaml
```

## License

MIT