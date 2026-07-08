# Model Benchmark Suite

A Rust benchmark suite for evaluating LLM models with **direct model execution**: launch each model, benchmark against its local API, then stop it.

Supports **MMLU-Pro**, **GPQA Diamond**, **AIME 2025/2026**, **MATH-500**, **Coding Eval** (HumanEval+, MBPP+), **SWE-Bench**, and **KLD divergence** between models.

## Quick Start

1. Configure your models in `models_config.yaml`
2. Set `HF_TOKEN` if using GPQA Diamond (gated dataset):
   ```bash
   export HF_TOKEN="hf_..."
   ```
3. Run benchmarks:
   ```bash
   cargo run -- run --config models_config.yaml
   ```

## Prerequisites

- **Rust 1.83+** (stable)
- A GGUF model and a runner (llama-server, ollama, etc.) exposing an OpenAI-compatible API
- **Docker** (optional) — required only for code and SWE-Bench evaluations

## Installation

```bash
cargo install --path .
```

## Configuration

Edit `models_config.yaml`. Each model is:

1. Started with `cmd` (shell command)
2. Benchmarked against `proxy` using the `model` name in API calls
3. Stopped with `cmd_stop` (optional, defaults to SIGTERM)
4. Results saved after each model for resumability

The `benchmarks` list controls which benchmarks to run (default: all registered). The `benchmark` section holds per-benchmark configuration.

```yaml
models:
  - display_name: "MyModel Q4"
    model: "model"
    cmd: "llama-server -m /path/to/model.Q4.gguf --port 28287"
    proxy: "http://localhost:28287/v1"
    cmd_stop: "pkill -9 llama-server"

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

benchmark:
  mmlu_pro:
    num_samples: 100
    # subjects: "biology,chemistry"   # null = all
  kld:
    num_prompts: 50
    prompt_source: mmlu
  gpqa:
    # num_samples: 10   # 198 questions total; requires HF_TOKEN
    # subjects: "physics"
  aime:
    # num_samples: 30   # all 30 problems
    year: "2025"        # "2025" or "2026"
  math500:
    # num_samples: 50   # all 500 problems
    # subjects: "algebra"
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
```

### Docker for Code Benchmarks

When running code benchmarks inside Docker, mounting `/var/run/docker.sock` grants host-level access — only do this in trusted environments. Use `docker.host_repo_path` to set the host-visible repository path so bind mounts resolve correctly.

### Comparison Reports

Define a `comparisons` section to auto-generate filtered reports for specific model groups:

```yaml
comparisons:
  - title: "Q4 vs Q5"
    models:
      - "MyModel Q4"
      - "MyModel Q5"
```

After a benchmark run or `cargo run -- report`, comparison reports (e.g., `q4-vs-q5.html`) are generated alongside the main report.

## Running Benchmarks

```bash
cargo run -- run --config models_config.yaml
```

The script starts each model, runs the configured benchmarks, stops the model, and saves results. Pairwise KLD is computed across all models after the run.

**Results** are saved to `benchmark_results/`:
- `results.json` — raw benchmark data
- `benchmark_report.md` — Markdown report
- `benchmark_report.html` — styled HTML report

**Resuming**: kill the process, rerun — it skips models already completed. Use `--no-resume` for a full re-run.

## Testing Models

Validate configurations without running full benchmarks:

```bash
cargo run -- test-models --config models_config.yaml
```

This starts each model, checks health, sends a test prompt, stops the model, and prints a PASS/FAIL summary.

## Generating Reports

From existing results, regenerate all reports (main + comparisons):

```bash
cargo run -- report
```

Regenerate only comparison reports:

```bash
cargo run -- compare --config models_config.yaml
```

Options: `--results` (results JSON path), `--output` (output directory), `--config` (comparison definitions).

## Benchmarks

### MMLU-Pro

Auto-downloaded from HuggingFace (`TIGER-Lab/MMLU-Pro`). Supports up to 10 options (A–J), few-shot CoT prompting, and per-subject accuracy reporting.

### KLD Divergence

Collects logprobs from all models on shared prompts (defaults to MMLU-Pro). Computes pairwise KL divergence at the end — lower values mean more similar output distributions.

### GPQA Diamond

198 graduate-level science multiple-choice questions (biology, chemistry, physics). Zero-shot chain-of-thought with A–D answer extraction. **Requires `HF_TOKEN`**.

Config options: `num_samples`, `subjects` (comma-separated categories).

### AIME

American Invitational Mathematics Examination. Competition-level math problems with integer answer extraction from `\boxed{}` notation. Supports **AIME 2025** and **AIME 2026** datasets from MathArena on HuggingFace.

Config options: `num_samples` (30 total), `year` ("2025" or "2026").

### MATH-500

500 competition-level math problems across 7 subjects. Zero-shot chain-of-thought with integer answer extraction.

Config options: `num_samples` (500 total), `subjects` (algebra, geometry, number theory, precalculus, etc.).

### Coding Eval

Docker-backed function-completion evaluation using **EvalPlus-style** oracle-separated testing:

- `humaneval_plus` — HumanEval+ (EvalPlus)
- `mbpp_plus` — MBPP+ (EvalPlus)
- `humaneval` — original HumanEval (public tests)

`enable_pass2`/`enable_pass3` enable iterative repair: the failed solution and error summary are fed back to the model. If an earlier attempt passes, later attempts are skipped.

Generated code is saved under `benchmark_results/coding_eval_runs/...`.

### SWE-Bench

Docker-backed repository patch benchmarks:

- `swebench` — Basic/full dataset
- `swebench_verified` — Verified
- `swebench_pro` — Pro-style gated dataset (requires `HF_TOKEN` and/or `dataset_id`)

Harness images auto-build from `docker/swebench-harness/Dockerfile` when `docker.build_images: true`. To build manually:

```bash
make swebench-harness-image
```

Predictions are saved under `benchmark_results/swe_bench_runs/...`. Runs can be slow due to repository-specific environment builds.

## Adding New Benchmarks

1. Create `src/benchmarks/my_bench.rs` implementing the `Benchmark` trait:

   ```rust
   pub struct MyBenchmark;

   impl Benchmark for MyBenchmark {
       fn name(&self) -> &str { "my_bench" }

       fn execute(&self, model: &Model, config: &serde_yaml::Value)
           -> Result<serde_json::Value>
       {
           // Your benchmark logic
           Ok(serde_json::json!({"my_metric": 0.8}))
       }
   }
   ```

2. Register in `src/benchmarks/mod.rs`:

   ```rust
   pub mod my_bench;
   // In registry():
   map.insert("my_bench".to_string(), Box::new(my_bench::MyBenchmark));
   ```

3. Add `"my_bench"` to the `benchmarks` list in your config.

## License

MIT