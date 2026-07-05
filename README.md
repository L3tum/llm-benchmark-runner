# Model Benchmark Suite

A modular Python benchmark suite for evaluating LLM models with **direct model execution**: launch each model, benchmark against its local API, then stop it. Supports **MMLU-Pro** (auto-downloaded) and **KLD divergence** between models.

## Features

- **Direct execution**: each model starts with a custom `cmd`, benchmarks run against its local OpenAI-compatible proxy, then `cmdStop` tears it down.
- **MMLU-Pro**: benchmark with up to 10 options per question, CoT few-shot prompting, per-subject accuracy.
- **KLD**: pairwise Kullback-Leibler divergence between model distributions on shared prompts.
- **Resumable**: results saved after each model; rerunning skips completed models.
- **Extensible benchmarks**: add a new benchmark by creating `benchmarks/<name>.py` with a `run_benchmark(model, config)` function.
- **CLI config override**: `--config path/to/config.yaml`.

## Prerequisites

- **Python 3.10+**
- A GGUF model and a runner (llama-server, ollama, etc.) that exposes an OpenAI-compatible API

## Installation

```bash
pip install -r requirements.txt
```

## Configuration

Edit `models_config.yaml`:

```yaml
models:
  - display_name: "MyModel Q4"
    cmd: "llama-server -m /path/to/model.Q4.gguf --port 28287"
    cmdStop: "pkill -9 llama-server"   # optional, defaults to SIGTERM
    proxy: "http://localhost:28287/v1"
  
  - display_name: "MyModel Q5"
    cmd: "llama-server -m /path/to/model.Q5.gguf --port 28288"
    proxy: "http://localhost:28288/v1"

num_samples: 100

mmlu_pro:
  num_samples: null  # overrides global if set
  subjects: null  # filter by subjects (comma-separated), null = all

kld:
  num_prompts: null  # overrides global if set
  prompt_source: "mmlu"  # uses MMLU-Pro questions for KLD prompts
  custom_prompts_path: null
```

Each model is:
1. Started with `cmd` (shell command)
2. Benchmark tests run against `proxy`
3. Stopped with `cmdStop` (optional)
4. Results saved after each model for resumability

## Running Benchmarks

```bash
python benchmark_runner.py --config models_config.yaml
```

The script will:
1. Start model 1, run MMLU-Pro and collect KLD logits, stop it, save results
2. Start model 2, run MMLU-Pro and collect KLD logits, stop it, save results
3. Compute pairwise KLD and generate reports

Results are saved to `benchmark_results/`:
- `results.json` — raw benchmark data
- `benchmark_report.md` — Markdown report
- `benchmark_report.html` — styled HTML report

**Resuming**: kill the process, rerun — it will skip models already completed.

## Testing Models (Without Running Benchmarks)

Before running full benchmarks, you can validate that each model's configuration works (start, health, prompt, stop):

```bash
python benchmark_runner.py test-models --config models_config.yaml
```

This command:
1. Starts each model with `cmd`
2. Waits for the proxy to become healthy
3. Sends a simple test prompt: "Say hello in one word."
4. Stops the model with `cmdStop` (or SIGTERM/SIGKILL)
5. Prints a summary table (PASS/FAIL) for each model

Useful for CI/CD pipelines or quickly verifying config changes without spending time on full benchmarks.

## Adding New Benchmarks

1. Create `benchmarks/my_bench.py` with:
   ```python
   def run_benchmark(model: dict, config: dict) -> dict:
       # model: {display_name, cmd, cmdStop, proxy, ...}
       # config: your benchmark-specific config section
       return {"my_metric": 0.8}
   ```
2. Register in `benchmarks/__init__.py`'s `BENCHMARKS` dict
3. Call it from `benchmark_runner.py` (or add a generic benchmark loop there)

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