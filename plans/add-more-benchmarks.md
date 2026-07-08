# Plan: Add More Benchmarks (Split into Two Phases)

## Context

The project is a benchmark runner that evaluates LLMs across multiple dimensions: reasoning (MMLU-Pro, GPQA, AIME, Math500), knowledge (KLD), long-context coding (SWE-Bench Verified, SWE-Bench Pro), short-context coding (HumanEval, HumanEval+, MBPP+), and creative generation (Minebench). All Docker-based benchmarks use `docker_runner.rs` for containerized evaluation.

The user wants to add:
- **Carwash benchmark** — common-sense physical reasoning
- **IFEval** — instruction following
- **HarmBench** — adversarial safety (with optional Docker-based extended execution)
- **DeepSWE** — long-horizon agentic coding (with Docker via `docker_runner.rs`)
- **Long-horizon context tasks** — general long-context evaluation

## Approach

Split the work into **two phases**:
- **Phase 1**: Easy, no-Docker benchmarks (Carwash, IFEval, HarmBench base)
- **Phase 2**: Docker-based benchmarks (DeepSWE, HELMET, HarmBench extended jailbreak) — each as its own sub-plan

All benchmarks follow the existing `Benchmark` trait pattern.

---

## Phase 1: Easy Benchmarks (No Docker Required) ✅ **COMPLETE**

### 1. Carwash Benchmark (`src/benchmarks/carwash.rs`)

**What it is**: A single-prompt sanity check for physical common-sense reasoning. The prompt asks: *"The car wash is 50 meters away from my current location. Should I walk or drive to the car wash?"*

**Category**: `Reasoning` (existing category).

**Implementation**: Simple `chat_completion` call with a pass/fail heuristic (check if response contains "drive" and does not contain "walk"). No Docker.

**Score**: Boolean pass/fail, output tokens.

**Why**: Fun, fast sanity check — ensures basic object-permanence reasoning isn't broken. Good quick pre-check.

### 2. IFEval (`src/benchmarks/ifeval.rs`)

**What it is**: Instruction-Following Evaluation from Google. Tests verifiable constraints: word count, keyword inclusion/exclusion, formatting rules, negative constraints.

**Category**: `InstructionFollowing` — new category.

**Implementation**: Download test set (HuggingFace/direct), run each prompt, and score with regex-based verifiers (no Docker).

**Score**: Instruction-following rate (%).

**Why**: Critical capability for production models. Complements reasoning benchmarks. Easy to add, objective scoring.

**Typical requirements**: 100-500 prompts, each with 1-3 verifiable instructions. Fast — runs in minutes.

### 3. HarmBench Base (`src/benchmarks/harmbench.rs`)

**What it is**: Adversarial safety evaluation from Center for AI Safety. Base mode tests direct harmful prompts without jailbreak attacks.

**Category**: `Safety` — new category.

**Implementation**:

- **Config object** (`HarmBenchConfig`) with key:
  ```rust
  pub enable_extended_execution: bool, // default: false
  ```
  When `false` (default), only run **base harm prompts**: direct harmful prompts, check if model refuses (no Docker needed).
  When `true`, activates **extended Docker-based jailbreak tests** (see Phase 2).

- Score is simply the base harm refusal rate (% of harmful prompts the model correctly refuses).

**Score**: Base harm refusal rate (%), output tokens.

**Why**: Essential safety evaluation. Base mode is fast and gives a good signal for production readiness. Extended mode (Phase 2) adds rigorous jailbreak testing.

**Typical requirements**: 510 harmful prompts, fast execution without Docker.

---

## Phase 2: Docker-Based Benchmarks (Individual Plans)

### A. HarmBench Extended — Jailbreak Harness (`src/benchmarks/harmbench.rs` extended)

**Status**: TODO — Phase 1 is ready; extended execution is a future addition.

**What it is**: Adversarial jailbreak attack pipeline using Docker containers for red-teaming tools (e.g., PAIR, GCG, GTA).

**Implementation**: When `enable_extended_execution: true`, run the same HarmBench harmful prompts but wrapped in jailbreak attack templates, then score attack success rate (ASR). Uses `DockerRunConfig` with a dedicated `harmbench-redteam` image.

**Score**: Jailbreak ASR (%), output tokens.

**Why**: The gold standard for safety evaluation. Without it, you only have base refusal rate; with it, you have a rigorous adversarial robustness metric.

**Typical requirements**: Significant compute for multiple attack strategies.

### B. DeepSWE (`src/benchmarks/deepswe.rs`)

**What it is**: Long-horizon, contamination-free software engineering benchmark. 113 tasks across 91 repos, 5 languages (TypeScript, Go, Python, JavaScript, Rust), with program-based verifiers.

**Category**: `LongContextCoding` — existing category, fits alongside SWE-bench.

**Implementation**: Docker container with the DeepSWE harness (similar to SWE-bench). Mount tasks directory, run agent execution, then verify outputs with programmatic verifiers.

**Score**: Resolution rate (%), output tokens, thinking tokens.

**Why**: Measures frontier coding agent capability with contamination-free tasks. Perfect companion to SWE-bench Verified — one is from real GitHub issues, the other is hand-written original tasks.

**Typical requirements**: DeepSWE Docker image, potentially long execution (1-2 hours per sample). Needs Docker socket for agent execution.

### C. HELMET (`src/benchmarks/helmet.rs`)

**What it is**: Princeton's comprehensive long-context benchmark. 7 task categories: RAG, in-context learning, summarization, re-ranking, instruction following, multilingual, code. Tests at 128K+ context lengths.

**Category**: `Research` (existing category — aligns with GPQA/AIME which are already Research).

**Implementation**:
- Download datasets (HuggingFace, 34GB of data across 7 categories)
- Run each task with the model's full context window
- Score with task-specific evaluators (some may need Docker for execution environments, e.g., code evaluation, math)

**Score**: Per-category accuracy/F1, overall average.

**Why**: The most thorough long-context evaluation available. Shows that synthetic NIAH tests don't predict real downstream performance. Covers diverse real-world use cases.

**Typical requirements**: Large context window (128K+), multiple datasets (34GB total). Some tasks may need Docker execution environments similar to SWE-bench.

---

## Files to Modify (All Phases)

1. **`src/benchmarks/mod.rs`** — register new benchmarks (Phase 1: Carwash, IFEval, HarmBench base; Phase 2: DeepSWE, HELMET, HarmBench extended)
2. **`src/benchmarks/carwash.rs`** — new (Phase 1)
3. **`src/benchmarks/ifeval.rs`** — new (Phase 1)
4. **`src/benchmarks/harmbench.rs`** — new with `HarmBenchConfig { enable_extended_execution: bool }` (Phase 1: base; Phase 2: extended)
5. **`src/benchmarks/deepswe.rs`** — new (Phase 2)
6. **`src/benchmarks/helmet.rs`** — new (Phase 2)
7. **`docker/harmbench-redteam/`** — Docker image for red-teaming (PAIR/GCG/GTA or official HarmBench) — Phase 2
8. **`docker/deepswe-harness/`** — Docker image for DeepSWE execution (mini-swe-agent) — Phase 2
9. **`docker/helmet-harness/`** — Docker image for HELMET execution environments — Phase 2
10. **`src/config.rs`** — if new benchmark categories are needed (InstructionFollowing, Safety)
11. **`models_config.yaml`** — example configurations for new benchmarks

## Verification

### Phase 1 (Completed)
1. ✅ **Carwash**: Implemented — simple prompt test with pass/fail heuristic.
2. ✅ **IFEval**: Implemented — downloads dataset, scores verifiable instructions with regex.
3. ✅ **HarmBench base**: Implemented — base harm prompts with refusal scoring, config object with `enable_extended_execution` (defaults to `false`).

### Phase 2 (Planned)
4. **HarmBench extended** (Phase 2): Verify jailbreak harness runs in Docker, attack success rate is calculated correctly.
5. **DeepSWE**: Run 1-2 samples with a known agent model, verify resolution rate calculation matches reference.
6. **HELMET**: Test one category at a time (RAG, summarization) to verify scoring.

All Docker-based benchmarks follow the existing `DockerRunConfig` pattern with read-only filesystem, tmpfs, and timeout handling, as already implemented in `swe_bench.rs`.
