# Model Benchmark Suite

A Rust benchmark suite for evaluating LLM models with **direct model execution**: launch each model, benchmark against its local API, then stop it. Supports **MMLU-Pro** (auto-downloaded) and **KLD divergence** between models.

## Features

- **Direct execution**: each model starts with a custom `cmd`, benchmarks run against its local OpenAI-compatible proxy, then `cmd_stop` tears it down.
- **MMLU-Pro**: benchmark with up to 10 options per question, CoT few-shot prompting, per-subject accuracy.
- **KLD**: pairwise Kullback-Leibler divergence between model distributions on shared prompts.
- **Resumable**: results saved after each model; rerunning skips completed models.
- **Extensible benchmarks**: add a new benchmark by creating a module in `src/benchmarks/` and registering it in `src/benchmarks/mod.rs`.
- **CLI config override**: `--config path/to/config.yaml` for all commands.

## Prerequisites

- **Rust 1.83+** (stable)
- A GGUF model and a runner (llama-server, ollama, etc.) that exposes an OpenAI-compatible API

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

benchmark:
  mmlu_pro:
    num_samples: 100
    # subjects: null  # filter by subjects (comma-separated), null = all
  kld:
    num_prompts: 50
    prompt_source: mmlu
    # custom_prompts_path: null
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

## License

MIT