# Plan: Harden Coding Eval, Refactor Docker Execution, and Add SWE-Bench MVP

## Context

The last commit added `coding_eval`, a Docker-backed Python coding benchmark. Review found three concrete fixes and one larger architecture opportunity:

- EvalPlus-style tasks currently run the candidate and reference oracle in the same Python process, which can leak the reference to generated code and invalidate scores.
- `runner::run_model` can leak a started model process if setup fails before the explicit stop path.
- Coding taskset downloads cache response bodies without checking HTTP status.
- The Docker runner should become a reusable foundation for richer code/repo benchmarks.
- Add a runnable SWE-Bench MVP, including automatic dataset download for SWE-Bench Verified/Pro-style datasets.

## Approach

Recommended approach: implement this as one project with three phases in the same PR/branch:

1. **Safety fixes**: close the EvalPlus oracle leak, fix model-process cleanup, and harden downloads.
2. **Architecture refactor**: extract a reusable Docker execution module so `coding_eval` and `swe_bench` do not each construct ad-hoc `docker run` commands.
3. **SWE-Bench MVP**: add first-class SWE-Bench benchmark names that generate model patches, download datasets automatically, and evaluate predictions through a Dockerized official-harness-style workflow.

For EvalPlus/HumanEval+, replace the current in-process reference harness with an oracle-separated evaluator. The candidate container should receive only the candidate solution and one test input at a time; expected outputs/reference execution must happen outside that candidate process and outside its mounted filesystem. This keeps compatibility with EvalPlus-style task JSON while removing the direct reference oracle leak.

For benchmark configuration, simplify the user-facing shape: each dataset is a benchmark name. Instead of nesting `tasksets` under `coding_eval`, expose names like `humaneval`, `humaneval_plus`, `mbpp_plus`, `swebench`, `swebench_verified`, and `swebench_pro` in the top-level `benchmarks` list. Internally, these can share implementation modules and map to dataset metadata.

For SWE-Bench, add separate benchmark names rather than a large `swe_bench.datasets` list. The current Docker runner enables it, but SWE-Bench needs repository checkout/build/test orchestration, dataset-specific instance metadata, generated patch files, and official result parsing. Use the extracted Docker runner to launch a Dockerized SWE-Bench harness, but keep SWE-Bench-specific logic in its own module.

## Files to modify

- `src/benchmarks/coding_eval.rs` — replace unsafe EvalPlus harness, improve downloads, move Docker execution out if extracted.
- `src/runner.rs` — make model process lifecycle exception-safe.
- `src/benchmarks/mod.rs` — register extracted modules and the new SWE-Bench benchmark.
- `src/report.rs` and `templates/report.html` — keep coding report working and add SWE-Bench reporting.
- `src/config.rs` — add a top-level Docker/evaluation runtime config block shared by code benchmarks.
- `models_config.yaml`, `test_config.yaml`, `README.md` — document simplified benchmark names and shared Docker runtime config.
- Potential new files:
  - `src/docker_runner.rs` or `src/benchmarks/docker_runner.rs`
  - `src/benchmarks/coding_eval/{mod.rs,config.rs,harness.rs,datasets.rs}`
  - `src/benchmarks/swe_bench/{mod.rs,config.rs,datasets.rs,prompts.rs,evaluator.rs}`
  - `docker/swebench-harness.Dockerfile` or equivalent harness image definition, if no suitable prebuilt image is assumed

## Reuse

- Reuse `Benchmark` trait and registry in `src/benchmarks/mod.rs`.
- Reuse existing model start/health/stop flow in `src/runner.rs`, after adding lifecycle guard.
- Reuse `Client::chat_completion` in `src/client.rs` for code/patch generation.
- Reuse Docker command construction ideas from `run_in_docker` in `src/benchmarks/coding_eval.rs`.
- Reuse report extraction/rendering patterns from `src/report.rs` and `templates/report.html`.
- Reuse existing result directories under `benchmark_results/`, but separate function-run artifacts from repository-run artifacts.
- Reuse HuggingFace download patterns conceptually from GPQA/AIME/MATH-style benchmark loaders, but implement paged JSON row download/cache for SWE-Bench datasets.

## Steps

- [ ] Add an RAII process guard around model child processes, or validate all fallible setup before `start_model`, so model processes are stopped on every early return.
- [ ] Harden taskset download: use HTTP `error_for_status`, write to a temporary file, then atomically rename into cache.
- [ ] Add a backwards-compatible top-level `docker` config block in `src/config.rs` with defaults so existing config files continue to parse.
- [ ] Extract Docker execution into a reusable runner type with command config, mounts, timeout, stdout/stderr capture, container cleanup, and host path mapping.
- [ ] Refactor `coding_eval` into smaller modules and split dataset selection from implementation:
  - [ ] Keep `coding_eval` as a backwards-compatible alias if practical.
  - [ ] Register first-class function-completion benchmark names: `humaneval`, `humaneval_plus`, and `mbpp_plus`.
  - [ ] Map each benchmark name to its task type, default dataset URL, prompt style, and evaluator behavior internally.
- [ ] Replace the unsafe EvalPlus in-process reference harness with an oracle-safe evaluator:
  - [ ] For EvalPlus tasks, compute reference outputs in a separate trusted process/container or in Rust-controlled setup, never in the candidate process.
  - [ ] Run candidate code with only `solution.py`, an input payload, and a small driver visible in its workdir.
  - [ ] Compare candidate output to expected output outside the candidate process.
  - [ ] Keep per-case/per-task timeout and Docker cleanup behavior.
- [ ] Add regression tests proving generated candidate code cannot access reference globals/source and cannot pass by calling the oracle.
- [ ] Keep original HumanEval support, but document that original HumanEval exposes public tests while EvalPlus-style evaluation is oracle-separated.
- [ ] Add SWE-Bench benchmarks using the extracted Docker runner.
  - [ ] Register `swebench`, `swebench_verified`, and `swebench_pro` as benchmark names.
  - [ ] Map benchmark names to dataset aliases internally: `swebench`/Basic -> `princeton-nlp/SWE-bench`, `swebench_verified` -> `princeton-nlp/SWE-bench_Verified`, and `swebench_pro` -> configurable Pro dataset ID.
  - [ ] Implement automatic dataset download/cache from HuggingFace datasets-server row APIs using those aliases, with per-benchmark `dataset_id` override when needed.
  - [ ] Support `HF_TOKEN` or a shared token env for gated/private SWE-Bench Pro-style datasets.
  - [ ] Store normalized dataset rows as JSONL under the benchmark cache to avoid repeated downloads.
  - [ ] Generate model predictions as official-style JSONL records: `instance_id`, `model_name_or_path`, `model_patch`.
  - [ ] Prompt the model to emit a unified diff only, using `problem_statement`, `hints_text`, repo name, and base commit metadata.
  - [ ] Evaluate predictions with a Dockerized SWE-Bench harness workflow rather than a bespoke ad-hoc `pytest` command, so environment/image handling stays compatible with SWE-Bench expectations.
  - [ ] Parse SWE-Bench harness outputs into benchmark JSON: resolved count, total instances, resolution rate, per-instance status, output/thinking tokens, and error summaries.
  - [ ] Add report section and summary metrics for SWE-Bench.
- [ ] Update README/config comments for Docker socket risk, safe EvalPlus evaluation, SWE-Bench requirements, dataset caching, and expected runtime/cost.

## Verification

- `cargo check --locked`
- `cargo test --locked`
- Unit tests for download failure handling and process guard behavior.
- Unit/integration tests for EvalPlus oracle isolation using a malicious candidate.
- Manual smoke test with a tiny local HumanEval-style JSONL taskset.
- Manual smoke test with a tiny EvalPlus-style JSONL taskset.
- One local miniature repo fixture with a known failing test and a mock SWE-Bench-style instance/prediction that fixes it.
- SWE-Bench dataset-download smoke test against a tiny `num_samples: 1` Verified run.
- SWE-Bench harness dry run or mocked harness parse test to validate report extraction without requiring a long full evaluation.

## Proposed simplified config shape

Recommended shape: keep dataset choice in the top-level `benchmarks` list and move Docker/evaluation runtime knobs into one shared config block.

```yaml
benchmarks:
  - humaneval_plus
  - mbpp_plus
  - swebench_verified
  - swebench_pro

# Shared runtime for benchmarks that execute generated code or repository tests.
docker:
  enabled: true
  host_repo_path: /host/path/to/llm-benchmark-runner   # only needed for Docker socket passthrough
  default_timeout_secs: 8
  images:
    python: python:3.12
    swebench_harness: llm-benchmark-runner/swebench-harness:latest
  build_images: true
  max_workers: 1

benchmark:
  humaneval_plus:
    num_samples: 10
    timeout_secs: 8
    enable_pass2: false
    enable_pass3: false
  mbpp_plus:
    num_samples: 10
  swebench_verified:
    num_samples: 10
    split: test
  swebench:
    num_samples: 10
    split: test
  swebench_pro:
    num_samples: 5
    split: test
    token_env: HF_TOKEN
    # dataset_id override allowed for gated/private naming differences:
    # dataset_id: SWE-bench/SWE-bench_Pro
```

Config notes:

- Dataset selection is expressed by benchmark names, not nested dataset lists.
- Docker image/host path settings live in one shared `docker` block instead of being duplicated under every benchmark.
- Per-benchmark sections remain small and follow existing project style: sample count, timeout/attempt overrides, token env, split, and optional dataset ID override.
- `coding_eval` can stay as a backwards-compatible umbrella alias for the old config shape, but docs should prefer the explicit benchmark names.
- Internal implementation can still share code: `humaneval_plus`, `humaneval`, and `mbpp_plus` use the function-completion evaluator; `swebench*` use the SWE-Bench evaluator.
