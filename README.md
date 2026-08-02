# Model Benchmark Suite

A Rust benchmark suite for evaluating LLM models with **direct model execution**: launch each model, benchmark against its local API, then stop it.

Supports **MMLU-Pro**, **GPQA Diamond**, **AIME 2025/2026**, **MATH-500**, **Coding Eval** (HumanEval+, MBPP+), **MultiPL-E** (30+ languages including C# and Rust), **SWE-Bench** (including Multilingual across 9 languages), **KLD divergence**, **Minebench**, **IFEval**, **HarmBench**, **TerminalBench 2.1**, **StableToolBench**, **Fictional Language Translation**, **Efficient Language Translation**, and more.

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
  fictional_language:
    # num_samples: 30  # 10 easy + 10 medium + 10 hard
  efficient_language:
    # num_samples: 30  # 10 easy + 10 medium + 10 hard
  humaneval_plus:
    num_samples: 10
    timeout_secs: 8
    enable_pass2: false
    enable_pass3: false
  mbpp_plus:
    num_samples: 10
  multipl_e:
    # Default: all 31 languages. Comment line below to run all, or uncomment to filter:
    # languages: [cs, rs]  # e.g. C# and Rust only
    # num_samples: 20  # per language
    # timeout_secs: 30
  swebench_verified:
    num_samples: 1
    split: test
    timeout_secs: 1800
  swebench_pro:
    num_samples: 1
    split: test
    timeout_secs: 1800
    token_env: HF_TOKEN
  swebench_multilingual:
    num_samples: 10
    split: test
    timeout_secs: 1800

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

All benchmarks are registered by name and can be listed with the `benchmarks` key in your config. **60 benchmarks** are available across 14 categories.

### Knowledge (9)

| Benchmark | Config Key | Description |
|---|---|---|
| GPQA | `gpqa` | 198 graduate-level science MC questions (biology, chemistry, physics). Zero-shot CoT, A–D extraction. Requires `HF_TOKEN`. |
| MMLU-Pro | `mmlu_pro` | Massive Multitask Language Understanding Pro. Up to 10 options (A–J), few-shot CoT, per-subject accuracy. |
| MMLU-Pro+ | `mmlu_pro_plus` | Extended MMLU-Pro variant with additional tasks. |
| MMLU-ProX | `mmlu_prox` | Multilingual MMLU-Pro evaluated across multiple languages. |
| NQ Open | `nq_open` | Natural Questions Open — open-domain QA from Google's dataset. |
| PopQA | `popqa` | Knowledge-based QA benchmark focused on popular entities. |
| SuperGPQA | `supergpqa` | Expanded GPQA variant with additional graduate-level science questions. |
| TriviaQA | `triviaqa` | Closed-book trivia questions from the TriviaQA dataset. |
| RACE | `race` | Reading comprehension benchmark from Chinese high school English exams. |

### Math (2)

| Benchmark | Config Key | Description |
|---|---|---|
| AIME | `aime` | American Invitational Mathematics Examination. Competition math with `\boxed{}` answer extraction. Supports 2025/2026. |
| MATH-500 | `math500` | 500 competition-level math problems across 7 subjects. Zero-shot CoT with answer extraction. |

### Short-Context Coding (6)

| Benchmark | Config Key | Description |
|---|---|---|
| Coding Eval | `coding_eval` | Legacy umbrella benchmark for coding evaluations. |
| HumanEval | `humaneval` | Original HumanEval benchmark — function completion from docstrings. Public tests only. |
| HumanEval+ | `humaneval_plus` | EvalPlus HumanEval with oracle-separated adversarial tests. Docker evaluation. |
| MBPP+ | `mbpp_plus` | EvalPlus MBPP with oracle-separated adversarial tests. Docker evaluation. |
| MultiPL-E | `multipl_e` | HumanEval translated to 30+ languages (C#, Rust, Go, Java, JS, Python, etc.). Docker evaluation. |
| CRUXEval | `cruxeval` | Code understanding: predict inputs/outputs or repair bugs from code snippets. |

### Long-Context Coding (4)

| Benchmark | Config Key | Description |
|---|---|---|
| SWE-Bench | `swebench` | Fix real GitHub issues by generating patches. Full dataset (may be slow). Docker required. |
| SWE-Bench Verified | `swebench_verified` | Verified subset of SWE-Bench. Recommended for faster runs. Docker required. |
| SWE-Bench Pro | `swebench_pro` | Pro-level SWE-Bench with gated dataset. Requires `HF_TOKEN`. Docker required. |
| SWE-Bench Multilingual | `swebench_multilingual` | 300 tasks across 9 languages (C, C++, Go, Java, JS/TS, PHP, Ruby, Rust) from 42 repos. |

### Creative (4)

| Benchmark | Config Key | Description |
|---|---|---|
| Minebench | `minebench` | Generate Minecraft-style voxel architectures from natural language descriptions. JSON output parsed for validity. |
| Minebench (Tools) | `minebench_tools` | Minebench with tool-calling interface. |
| SVG: Flamingo Moonwalking | `svg_moonwalk` | Generate SVG of a flamingo moonwalking on a beach. Visual code generation test. |
| SVG: Pelican Riding a Bike | `svg_bike` | Generate SVG of a pelican riding a bicycle. Visual code generation test. |

### Reasoning (7)

| Benchmark | Config Key | Description |
|---|---|---|
| BBH | `bbh` | Big-Bench Hard — logic and reasoning tasks including tracking objects, date understanding, boolean expressions, fallacies, and more. |
| Carwash | `carwash` | Common-sense reasoning sanity check: "should you walk or drive to the car wash?" |
| CRUXEval | `cruxeval` | Code reasoning: predict program inputs from outputs, outputs from inputs, or repair buggy code. |
| Efficient Language | `efficient_language` | Decode sentences from a compressed symbolic language. 3 difficulty levels. Tests contextual inference. |
| Fictional Language | `fictional_language` | Translate from a fictional language using provided vocabulary/grammar rules. 3 difficulty levels. |
| SNLI | `snli` | Stanford Natural Language Inference — determine if a hypothesis is entailed, contradicted, or neutral. |
| TruthfulQA (MC1) | `truthful_qa` | Multiple-choice truthfulness evaluation (single-answer variant). |

### Research (4)

| Benchmark | Config Key | Description |
|---|---|---|
| FactBench | `factbench` | In-the-wild factuality evaluation from real-world queries. |
| RULER | `ruler` | Retrieval and Long-context Evaluation — needle-in-haystack and retrieval benchmarks at varying context lengths. |
| SciFact | `scifact` | Scientific claim verification — judge whether evidence supports a scientific claim. |
| PopQA | `popqa` | Knowledge-based QA from Wikipedia-derived popular entities. |

### Similarity (1)

| Benchmark | Config Key | Description |
|---|---|---|
| KLD Divergence | `kld` | Pairwise KL divergence from logprobs on shared prompts. Lower = more similar output distributions. |

### Instruction Following (1)

| Benchmark | Config Key | Description |
|---|---|---|
| IFEval | `ifeval` | ~1,000 prompts with verifiable constraints (word count, keywords, formatting). Measures instruction adherence. |

### Hallucination (14)

| Benchmark | Config Key | Description |
|---|---|---|
| CNN/Daily Mail | `cnn_dailymail` | News summarization evaluated with ROUGE-based faithfulness proxy. |
| FEVER | `fever` | Fact Extraction and VERification — classify claims as SUPPORTS/REFUTES/NOT ENOUGH INFO. |
| FaithDial | `faithdial` | Faithfulness evaluation for dialog summarization. |
| HaluBench | `halubench` | RAG hallucination detection across multiple knowledge domains. |
| HaluEval | `halueval` | Hallucination evaluation via QA on model-generated text. |
| HDM-Bench | `hdm_bench` | Hallucination detection in multi-document settings. |
| SQuAD 2.0 | `squad_v2` | Stanford QA dataset v2 with unanswerable questions. Tests both answer accuracy and refusal behavior. |
| Tool Hallucination | `tool_hallucination` | Tests whether models truthfully report tool failures across varying numbers of tool calls. |
| True-False | `true_false` | Binary fact verification — determine if statements are true or false. |
| TruthfulQA (MC2) | `truthful_qa_mc2` | Multiple-choice truthfulness evaluation (multi-answer variant). |
| TruthfulQA Gen | `truthful_qa_gen` | Open-ended truthfulness generation — evaluates model responses against human-written truthful answers. |
| XSum | `xsum` | Abstractive summarization evaluated with ROUGE-based faithfulness proxy from BBC News articles. |
| Race | `race` | Reading comprehension evaluated for hallucination in reading comprehension tasks. |
| TriviaQA | `triviaqa` | Knowledge QA evaluated for factual hallucination. |

### Translation (1)

| Benchmark | Config Key | Description |
|---|---|---|
| EA-MT | `ea_mt` | Entity-Aware Machine Translation — evaluates how well models translate multilingual sentences while preserving named entities. |

### Safety (1)

| Benchmark | Config Key | Description |
|---|---|---|
| HarmBench | `harmbench` | Safety evaluation. Sends ~500 harmful behavior prompts as "How to \<goal\>". Measures refusal rate. |

### Tool Use (2)

| Benchmark | Config Key | Description |
|---|---|---|
| StableToolBench | `stable_toolbench` | Tool/API selection and parameter completion from ~1.2K instances across 6 subsets. |
| TerminalBench 2.1 | `terminal_bench` | Agentic terminal tasks — model executes shell commands iteratively via tool calls to complete tasks. |

### String Manipulation (9)

| Benchmark | Config Key | Description |
|---|---|---|
| Base64 | `base64` | Encode/decode text in Base64 format. |
| Base64 (Tools) | `base64_tools` | Base64 encoding/decoding with tool-calling interface. |
| Hex | `hex` | Encode/decode text in hexadecimal format. |
| Hex (Tools) | `hex_tools` | Hex encoding/decoding with tool-calling interface. |
| Morse Code | `morse_code` | Encode text to Morse code or decode Morse code to text. |
| Morse Code (Tools) | `morse_code_tools` | Morse code translation with tool-calling interface. |
| Reverse Writing | `reverse` | Reverse the characters of a given word/string. |
| Reverse Writing (Tools) | `reverse_tools` | Reverse writing with tool-calling interface. |

---

### Detailed Benchmark Guides

#### MMLU-Pro (`mmlu_pro`)

Auto-downloaded from HuggingFace (`TIGER-Lab/MMLU-Pro`). Supports up to 10 options (A–J), few-shot chain-of-thought prompting, and per-subject accuracy reporting.

**Config options:** `num_samples`, `subjects` (comma-separated subjects, `null` = all).

#### KLD Divergence (`kld`)

Collects logprobs from all models on shared prompts. Computes pairwise KL divergence at the end — lower values mean more similar output distributions.

**Config options:** `num_prompts`, `prompt_source` (e.g., `mmlu` to use MMLU-Pro prompts), `custom_prompts_path` (path to a prompts file).

#### GPQA Diamond (`gpqa`)

198 graduate-level science multiple-choice questions (biology, chemistry, physics). Zero-shot chain-of-thought with A–D answer extraction. **Requires `HF_TOKEN`**.

**Config options:** `num_samples`, `subjects` (comma-separated: biology, chemistry, physics).

#### AIME (`aime`)

American Invitational Mathematics Examination. Competition-level math problems with integer answer extraction from `\boxed{}` notation. Supports **AIME 2025** and **AIME 2026** datasets from MathArena on HuggingFace.

**Config options:** `num_samples` (30 total), `year` ("2025" or "2026").

#### MATH-500 (`math500`)

500 competition-level math problems across 7 subjects (algebra, geometry, number theory, precalculus, probability, counting & combinatorics, intermediate algebra). Zero-shot chain-of-thought with integer answer extraction.

**Config options:** `num_samples` (500 total), `subjects` (comma-separated subject names).

#### Minebench (`minebench`)

3D voxel building task from the Minebench benchmark. Models are prompted to generate Minecraft-style voxel architectures. Builds are parsed from JSON output (boxes, lines, blocks) and evaluated for validity and completeness.

**Config options:** `buildings` (list of predefined building prompts like `castle`, `dragon`, `train`), `build` (single custom build description).

#### Carwash (`carwash`)

A simple common-sense reasoning sanity check. Sends a single "drive vs walk to car wash" prompt and checks that the model's answer is logically correct.

No config options — runs with defaults.

#### Fictional Language Translation (`fictional_language`)

Evaluates **reasoning** by asking the model to translate sentences from a fictional language to English. Each instance provides a vocabulary, grammar rules (possibly partial), and primer examples. The model must then translate test sentences from the fictional language.

**Three difficulty levels**, each scored independently plus an overall score:
- **Easy**: Full vocabulary and grammar rules provided. Model applies known rules.
- **Medium**: Full vocabulary but partial grammar rules. Model infers missing grammar from examples.
- **Hard**: Partial vocabulary and partial grammar. Model infers both vocabulary and grammar from context.

~30 instances total (10 per difficulty level). No Docker required.

**Config options:** `num_samples` (default: all 30).

#### Efficient Language Translation (`efficient_language`)

Evaluates **reasoning** by asking the model to decode sentences written in a highly compressed symbolic language back to English. Each instance provides a symbol dictionary (possibly partial) and encoded sentences. The model must substitute symbols with their meanings.

**Three difficulty levels**, each scored independently plus an overall score:
- **Easy**: Complete symbol dictionary (100% of symbols defined). Model just looks up.
- **Medium**: Partial dictionary (~60-70% defined). Model infers missing symbols from context.
- **Hard**: Sparse dictionary (~30-40% defined) with compound expressions. Model infers symbols and composition rules.

~30 instances total (10 per difficulty level). No Docker required.

**Config options:** `num_samples` (default: all 30).

#### MultiPL-E (`multipl_e`)

**Multi-language coding benchmark** — translates the HumanEval benchmark (164 function-completion tasks) to **30+ programming languages**. The model is given a function signature and docstring translated into the target language, and must generate the function body. Solutions are evaluated by compiling/running the code in Docker containers.

**Supported languages** include: C# (`cs`), Rust (`rs`), Go, Java, JavaScript, TypeScript, C++, Python, Ruby, PHP, Haskell, Swift, Scala, Shell, Ada, Clojure, Coq, D, Dafny, Dart, Elixir, F#, Julia, Lean, Lua, Luau, OCaml, Perl, R, Racket, MATLAB, and more.

Default languages: **all 31 supported languages**. Filter to specific ones via config if desired.

Requires **Docker** with the `ghcr.io/nuprl/multipl-e-evaluation` image.

**Config options:** `languages` (list of language codes, default: all available), `num_samples` (per language), `timeout_secs` (per-task evaluation timeout, default: 30).

#### IFEval (`ifeval`)

**Instruction-Following Evaluation** from Google. ~1,000 prompts with verifiable constraints (word count, keyword rules, formatting requirements, etc.). Downloads from HuggingFace, runs fast, no Docker required. Measures how often the model correctly follows all instructions.

No config options.

#### FEVER (`fever`)

**Fact Extraction and VERification** from the FEVER dataset. Evaluates whether a model can correctly classify claims as SUPPORTS, REFUTES, or NOT ENOUGH INFO using 16-shot prompting. The benchmark tries multiple download sources (official website, HuggingFace, GitHub mirror) with automatic fallback. If all automatic downloads fail, it provides instructions for manual download and caching.

No config options.

#### HarmBench (`harmbench`)

Safety evaluation (base mode). Downloads the HarmBench "danger" dataset (~500 harmful behaviors) and sends them as "How to \<goal\>" prompts. Measures how often the model refuses. Does not require Docker.

**Config options:** `enable_extended_execution` (future: extended jailbreak harness, currently disabled).

#### Coding Eval (HumanEval+, MBPP+, HumanEval)

Docker-backed function-completion evaluation using **EvalPlus-style** oracle-separated testing. Code is sent to the model, generated solutions are tested in isolated Docker containers.

- **`humaneval_plus`** — HumanEval+ (EvalPlus, includes adversarial tests)
- **`mbpp_plus`** — MBPP+ (EvalPlus, includes adversarial tests)
- **`humaneval`** — original HumanEval (public tests only)

`enable_pass2`/`enable_pass3` enable iterative repair: the failed solution and error summary are fed back to the model. If an earlier attempt passes, later attempts are skipped.

**Config options:** `num_samples`, `timeout_secs`, `enable_pass2`, `enable_pass3`, `language_images` (custom Docker image per language).

Generated code is saved under `benchmark_results/coding_eval_runs/...`.

#### SWE-Bench

Docker-backed repository patch benchmarks. Models are given a bug report and a repository state, and must generate a patch that fixes the issue. The harness validates patches by applying them and running the repository's test suite.

- **`swebench`** — Full dataset (may be slow)
- **`swebench_verified`** — Verified subset (recommended)
- **`swebench_pro`** — Pro-style gated dataset (requires `HF_TOKEN` and/or `dataset_id`)
- **`swebench_multilingual`** — 300 tasks across **9 programming languages** (C, C++, Go, Java, JavaScript/TypeScript, PHP, Ruby, Rust) from 42 repositories. Uses the same harness as SWE-Bench but provides cross-language evaluation. Includes per-language breakdown in the report.

Harness images auto-build from `docker/swebench-harness/Dockerfile` when `docker.build_images: true`. To build manually:

```bash
make swebench-harness-image
```

Predictions are saved under `benchmark_results/swe_bench_runs/...`. Runs can be slow due to repository-specific environment builds.

**Config options:** `num_samples`, `split` (e.g., `test`), `timeout_secs` (per-task timeout), `token_env` (env var for API key), `dataset_id` (HuggingFace dataset for pro version).

#### TerminalBench 2.1 (`terminal_bench`)

Docker-backed **agentic terminal** benchmark. Each task provides a Linux shell environment with an instruction (e.g., "find all files containing a pattern", "debug a failing script"). The model acts as a terminal agent, executing shell commands iteratively via structured tool calls until the task is complete. Tasks are validated by running the repository's test suite inside the container.

Dataset downloaded from `harbor-framework/terminal-bench-2-1` on GitHub and cached locally.

**Config options:** `num_samples` (89 total), `max_iterations` (max tool-call turns per task, default 50), `timeout_secs` (per-task timeout, default 900), `categories` (filter by category: `file_management`, `shell_commands`, `debugging`, `programming`, `data_processing`, `linux`, `other`).

#### StableToolBench (`stable_toolbench`)

Evaluates **tool/API selection and parameter completion** from the THUNLP-MT/StableToolBench dataset. Each instance presents a natural-language query along with a catalog of available APIs. The model is expected to select the correct tool(s) and provide appropriate parameters via structured function calling. Metrics include simulated pass rate, tool selection accuracy, API precision/recall/F1, and parameter completeness.

Dataset downloaded directly from `THUNLP-MT/StableToolBench/solvable_queries/` on GitHub (~1.2K instances across 6 subsets: G1_instruction, G1_category, G1_tool, G2_category, G2_instruction, G3_instruction).

**Config options:** `num_samples`, `subsets` (comma-separated subset names), `categories` (filter by domain).

## Legacy: `coding_eval` (umbrella)

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