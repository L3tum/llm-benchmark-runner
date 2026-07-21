# Model Benchmark Suite

A Rust benchmark suite for evaluating LLM models with **direct model execution**: launch each model, benchmark against its local API, then stop it.

Supports **MMLU-Pro**, **GPQA Diamond**, **AIME 2025/2026**, **MATH-500**, **Coding Eval** (HumanEval+, MBPP+), **SWE-Bench**, **KLD divergence**, **Minebench**, **IFEval**, **HarmBench**, and more.

## Quick Start

1. Configure your models in `models_config.yaml`
2. Set `HF_TOKEN` if using gated datasets (GPQA, SWE-bench-pro):
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

### Official Minebench Renderer (Optional)

For full-fidelity Minecraft-like voxel rendering using the official [Ammaar-Alam/minebench](https://github.com/Ammaar-Alam/minebench) renderer:

1. Install **Node.js 18+** and npm (https://nodejs.org/)
2. Build with the `renderer-official` feature:
   ```bash
   cargo build --release --features renderer-official
   ```
   This downloads the official TypeScript source, compiles it with esbuild, and embeds the renderer alongside the pre-built Three.js.
3. The generated report will use the official renderer with texture atlas instead of the default color-palette renderer.

The default build (`cargo build`) uses a lightweight custom renderer with no extra dependencies.

### Docker (Default — with Official Minebench Renderer)

The Docker build **always includes the official renderer** (Node.js is installed during the build step automatically):

```bash
docker build -t llm-benchmark-runner .
# or
make docker
```

This produces an image with the full-fidelity texture-atlas voxel renderer.

## Configuration

Edit `models_config.yaml`. Each model is:

1. Started with `cmd` (shell command)
2. Benchmarked against `proxy` using the `model` name in API calls
3. Stopped with `cmd_stop` (optional, defaults to SIGTERM)
4. Results saved after each model for resumability

The `benchmarks` list controls which benchmarks to run (default: all registered). The `benchmark` section holds per-benchmark configuration.

### Macros & Variables (Config Reuse)

Like llama-swap's macros, you can define **variables** and **block-level templates** (`!macro`) to avoid repetition across configs.

**Variables** — define in `variables:` at the top, use `{{name}}` inline anywhere:

```yaml
variables:
  llama-server: "llama-server"
  model-dir: "/models"
  common-flags: "--threads 8"
  base-port: 28287

models:
  - display_name: "Q4 Model"
    model: "model"
    cmd: "{{llama-server}} -m {{model-dir}}/model.Q4.gguf {{common-flags}} --port {{base-port}}"
    proxy: "http://localhost:{{base-port}}/v1"
```

**Macros** — define block templates in `macros:` and instantiate with `!macro [name, {args}]`:

```yaml
macros:
  server_model:
    model_name: "model"
    cmd: "{{llama-server}} -m {{model-dir}}/model.{{variant}}.gguf {{common-flags}} --port {{port}}"
    proxy: "http://localhost:{{port}}/v1"

models:
  - !macro [server_model, {variant: Q4, port: 28287}]
    display_name: "Q4 Model"
  - !macro [server_model, {variant: Q5, port: 28288}]
    display_name: "Q5 Model"
```

**How it works:**
- Macro arguments act as local variables that override global variables for that block
- Macros can nest — a macro can call another macro (with cycle detection)
- Standard YAML anchors (`&`/`*`) still work alongside macros

A full example config is in `test_macro_config.yaml`.

**Nested Macros** — a macro template can invoke another macro as a field value:

```yaml
macros:
  outer:
    msg: !macro [inner, {name: World}]
    extra: "field"
  inner:
    msg: "Hello {{name}}!"

result:
  !macro [outer, {}]
```

The `!macro` call must be the value of a key (not a bare block scalar), so YAML remains valid. The expanded `result` becomes:

```yaml
result:
  msg: "Hello World!"
  extra: "field"
```

### Full Configuration Example

```yaml
variables:
  llama-server: "llama-server"
  model-dir: "/models"
  common-flags: "--threads 8"

macros:
  server_model:
    model_name: "model"
    cmd: "{{llama-server}} -m {{model-dir}}/model.{{variant}}.gguf {{common-flags}} --port {{port}}"
    proxy: "http://localhost:{{port}}/v1"

models:
  - !macro [server_model, {variant: Q4, port: 28287}]
    display_name: "Q4 Model"
  - !macro [server_model, {variant: Q5, port: 28288}]
    display_name: "Q5 Model"

benchmarks:
  - mmlu_pro
  - kld
  - gpqa
  - aime
  - math500
  - minebench
  - carwash
  - ifeval
  - harmbench
  - humaneval_plus
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
  minebench:
    # buildings: [castle, dragon]
    # build: "A compact test castle"
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

# Comparison groups for generating filtered reports
comparisons:
  - title: "Q4 vs Q5"
    models:
      - "Q4 Model"
      - "Q5 Model"
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

## Available Benchmarks

All benchmarks are registered by name and can be listed with the `benchmarks` key in your config.

### MMLU-Pro (`mmlu_pro`)

Auto-downloaded from HuggingFace (`TIGER-Lab/MMLU-Pro`). Supports up to 10 options (A–J), few-shot chain-of-thought prompting, and per-subject accuracy reporting.

**Config options:** `num_samples`, `subjects` (comma-separated subjects, `null` = all).

### KLD Divergence (`kld`)

Collects logprobs from all models on shared prompts. Computes pairwise KL divergence at the end — lower values mean more similar output distributions.

**Config options:** `num_prompts`, `prompt_source` (e.g., `mmlu` to use MMLU-Pro prompts), `custom_prompts_path` (path to a prompts file).

### GPQA Diamond (`gpqa`)

198 graduate-level science multiple-choice questions (biology, chemistry, physics). Zero-shot chain-of-thought with A–D answer extraction. **Requires `HF_TOKEN`**.

**Config options:** `num_samples`, `subjects` (comma-separated: biology, chemistry, physics).

### AIME (`aime`)

American Invitational Mathematics Examination. Competition-level math problems with integer answer extraction from `\boxed{}` notation. Supports **AIME 2025** and **AIME 2026** datasets from MathArena on HuggingFace.

**Config options:** `num_samples` (30 total), `year` ("2025" or "2026").

### MATH-500 (`math500`)

500 competition-level math problems across 7 subjects (algebra, geometry, number theory, precalculus, probability, counting & combinatorics, intermediate algebra). Zero-shot chain-of-thought with integer answer extraction.

**Config options:** `num_samples` (500 total), `subjects` (comma-separated subject names).

### Minebench (`minebench`)

3D voxel building task from the Minebench benchmark. Models are prompted to generate Minecraft-style voxel architectures. Builds are parsed from JSON output (boxes, lines, blocks) and evaluated for validity and completeness.

**Config options:** `buildings` (list of predefined building prompts like `castle`, `dragon`, `train`), `build` (single custom build description).

### Carwash (`carwash`)

A simple common-sense reasoning sanity check. Sends a single "drive vs walk to car wash" prompt and checks that the model's answer is logically correct.

No config options — runs with defaults.

### IFEval (`ifeval`)

**Instruction-Following Evaluation** from Google. ~1,000 prompts with verifiable constraints (word count, keyword rules, formatting requirements, etc.). Downloads from HuggingFace, runs fast, no Docker required. Measures how often the model correctly follows all instructions.

No config options.

### FEVER (`fever`)

**Fact Extraction and VERification** from the FEVER dataset. Evaluates whether a model can correctly classify claims as SUPPORTS, REFUTES, or NOT ENOUGH INFO using 16-shot prompting. The benchmark tries multiple download sources (official website, HuggingFace, GitHub mirror) with automatic fallback. If all automatic downloads fail, it provides instructions for manual download and caching.

No config options.

Safety evaluation (base mode). Downloads the HarmBench "danger" dataset (~500 harmful behaviors) and sends them as "How to \<goal\>" prompts. Measures how often the model refuses. Does not require Docker.

**Config options:** `enable_extended_execution` (future: extended jailbreak harness, currently disabled).

### Coding Eval (HumanEval+, MBPP+, HumanEval)

Docker-backed function-completion evaluation using **EvalPlus-style** oracle-separated testing. Code is sent to the model, generated solutions are tested in isolated Docker containers.

- **`humaneval_plus`** — HumanEval+ (EvalPlus, includes adversarial tests)
- **`mbpp_plus`** — MBPP+ (EvalPlus, includes adversarial tests)
- **`humaneval`** — original HumanEval (public tests only)

`enable_pass2`/`enable_pass3` enable iterative repair: the failed solution and error summary are fed back to the model. If an earlier attempt passes, later attempts are skipped.

**Config options:** `num_samples`, `timeout_secs`, `enable_pass2`, `enable_pass3`, `language_images` (custom Docker image per language).

Generated code is saved under `benchmark_results/coding_eval_runs/...`.

### SWE-Bench

Docker-backed repository patch benchmarks. Models are given a bug report and a repository state, and must generate a patch that fixes the issue. The harness validates patches by applying them and running the repository's test suite.

- **`swebench`** — Full dataset (may be slow)
- **`swebench_verified`** — Verified subset (recommended)
- **`swebench_pro`** — Pro-style gated dataset (requires `HF_TOKEN` and/or `dataset_id`)

Harness images auto-build from `docker/swebench-harness/Dockerfile` when `docker.build_images: true`. To build manually:

```bash
make swebench-harness-image
```

Predictions are saved under `benchmark_results/swe_bench_runs/...`. Runs can be slow due to repository-specific environment builds.

**Config options:** `num_samples`, `split` (e.g., `test`), `timeout_secs` (per-task timeout), `token_env` (env var for API key), `dataset_id` (HuggingFace dataset for pro version).

### Legacy: `coding_eval` (umbrella)

Backwards-compatible config shape. If you list `coding_eval` in your `benchmarks` and configure a `tasksets` under `benchmark.coding_eval`, it will run the specified coding benchmarks. Prefer using the explicit benchmark names above.

## Adding New Benchmarks

1. Create `src/benchmarks/my_bench.rs` implementing the `Benchmark` trait:

   ```rust
   pub struct MyBenchmark;

   impl Benchmark for MyBenchmark {
       fn name(&self) -> &str { "my_bench" }

       fn execute(&self, model: &Model, config: &yaml_serde::Value)
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