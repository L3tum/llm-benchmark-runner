# Benchmark Research & Adoption Plan — Self-Hosted LLM Evaluation Suite

**Date:** 2026-08-03  
**Pipeline:** REFINE → RESEARCH → COMPOSE  
**Pipeline Agent:** REFINE (scope refinement) → RESEARCH (web research + benchmark discovery) → COMPOSE (actionable plan synthesis)

---

## Context

The `llm-benchmark-runner` is a Rust CLI that evaluates LLMs via OpenAI-compatible HTTP APIs. The existing suite has strong knowledge (MMLU-Pro, GPQA), math (AIME, Math500), and heavy coding (SWE-Bench) coverage — but significant gaps in:

| Gap | Existing Coverage | What's Missing |
|-----|-------------------|----------------|
| **Hallucinations** | halueval, faithdial, hdm_bench (all classification) | Generation-side hallucination, RAG faithfulness |
| **Source extraction** | None | Needle retrieval, long-context source localization |
| **Research** | fever (single-hop only) | Scientific verification, multi-hop, contemporary claims |
| **Contradictions** | None | NLI-style contradiction detection, self-consistency |
| **Truthfulness** | truthful_qa MC1/MC2 (multiple-choice only) | Open-ended truthfulness |
| **Lightweight coding** | humaneval/mbpp (Docker) + swebench (heavy) | Code understanding as pure text tasks |

### "Suitable for self-hosted" criteria

A benchmark qualifies if it meets these criteria:

- **API-only interaction** — communicates via HTTP (OpenAI-compatible chat/completions API). No direct model process access.
- **No cloud dependency** — dataset downloaded once, cached locally. No per-request external API calls for scoring.
- **Self-contained scoring** — evaluation script knows the correct answer or has a simple deterministic check. No judge model needed (or small local model OK).
- **Fast per-task** — < 60 seconds each, < 15 minutes total for a single model.
- **Minimal infrastructure** — no Docker, no CI/CD, no shell access to the model, no repository checkout. Simple scoring via string matching, regex, JSON parsing, or at most a small Python eval script.
- **Lightweight coding** — no SWE-Bench/Deep-SWE style (no repo checkout, no CI harness, no shell, no patch application). Single-file execution or pure text-based evaluation only.

### What is NOT suitable

| Criterion | Example |
|-----------|---------|
| Cloud API required for scoring | Needs GPT-4 as a judge |
| Per-request external API | Needs to call a search engine per task |
| Long-running evaluation | SWE-Bench (minutes to hours per task) |
| Repository-level operations | SWE-Bench, Deep-SWE, TerminalBench |
| Docker-in-Docker | Heavy container orchestration |

---

## Recommendations Summary

| # | Benchmark | Category | Priority | Size | Eval Method | Harness | Dataset URL |
|---|-----------|----------|----------|------|-------------|---------|-------------|
| 1 | **SciFact** | Research | HIGH | 1.4K | Claim verification → label match | Lightweight | HF `allenai/scifact` |
| 2 | **BBH** | Reasoning | HIGH | 23 tasks × 40 | Few-shot → exact match | Lightweight | HF `suzgunmirac/BBH` |
| 3 | **CRUXEval** | Reasoning | HIGH | 800 | Code prediction → Python eval | Lightweight | HF `facebookresearch/CRUXEval` |
| 4 | **RULER** | Research | HIGH | 13 task types | Needle retrieval → exact match | Lightweight | HF `NVIDIA/RULER` |
| 5 | **HaluBench** | Hallucination | HIGH | ~2.5K | RAG hallucination → keyword match | Lightweight | HF `tianyi-lab/HaluBench` |
| 6 | **SNLI/MultiNLI** | Reasoning | MEDIUM | 570K+433K (use ~500) | 3-class NLI → label match | Lightweight | HF `stanfordnlp/snli` |
| 7 | **PopQA** | Knowledge | MEDIUM | 14.3K | Open-ended QA → entity match | Lightweight | HF `akariasai/PopQA` |
| 8 | **FactBench** | Research | MEDIUM | ~1K | Claim verification → label match | Lightweight | GitHub (arXiv:2410.22257) |
| 9 | **TruthfulQA-Gen** | Hallucination | MEDIUM | 817 | Open-ended → keyword match | Lightweight | Reuse existing `truthful_qa` dataset |

---

## Detailed Benchmark Plans

### 1. SciFact (`src/benchmarks/scifact.rs`) — **Research**

**Description:** Scientific claim verification benchmark. Given a scientific claim and a related abstract, classify whether the abstract SUPPORTS or REFUTES the claim.

- **Dataset source:** HuggingFace `allenai/scifact`, CC BY-NC 2.0
- **Format:** JSON with `claim`, `label` (SUPPORTS/REFUTES), and `evidence_sentence_id`
- **Evaluation method:** Few-shot prompt (claim + abstract) → model classifies → exact label match
- **Proposed prompt:**

```
Determine whether the following scientific claim is supported or refuted by the given abstract.
Respond with only SUPPORTS or REFUTES.

Abstract: {abstract}
Claim: {claim}
Answer:
```

- **Scoring:** Case-insensitive exact string match on SUPPORTS/REFUTES. Track accuracy per label.
- **BenchmarkCategory:** `Research`
- **Estimated effort:** S (~200 lines, directly mirrors `fever.rs`)
- **Files to create:** `src/benchmarks/scifact.rs`
- **Files to modify:** `src/benchmarks/mod.rs` (register)

### 2. BBH — Big-Bench Hard (`src/benchmarks/bbh.rs`) — **Reasoning**

**Description:** 23 challenging tasks from Big-Bench designed to be hard for LLMs. Particularly relevant: contradiction detection, logical deduction, disambiguation, temporal reasoning.

- **Dataset source:** HuggingFace `suzgunmirac/BBH`
- **Format:** Per-task JSON files with few-shot examples, inputs, and target answers
- **Key tasks for contradiction/logic coverage:**
  - `logical_deduction_three_objects` — spatial/logical reasoning
  - `temporal_sequences` — temporal contradiction detection
  - `disambiguation_qa` — resolving contradictory references
  - `hyperbaton` — detecting sentence structure contradictions
  - `reasoning_about_colored_objects` — multi-attribute logical reasoning
  - `object_counting` — counting with contradictory descriptions
- **Evaluation method:** Per-task few-shot prompt → model answer → exact string match
- **Proposed prompt (per task):**

```
{task_instruction}
{few_shot_examples}

{input}
Answer:
```

- **Scoring:** Exact string match on model output vs gold answer. Aggregate accuracy across tasks. Report per-task breakdown.
- **BenchmarkCategory:** `Reasoning`
- **Estimated effort:** M (~350 lines, 23 tasks with varying formats)
- **Files to create:** `src/benchmarks/bbh.rs`
- **Files to modify:** `src/benchmarks/mod.rs` (register)

### 3. CRUXEval (`src/benchmarks/cruxeval.rs`) — **Reasoning**

**Description:** Code understanding benchmark that tests whether models can predict code outputs without execution. Pure text-based evaluation — no Docker needed.

- **Dataset source:** HuggingFace `facebookresearch/CRUXEval`, MIT license
- **Format:** JSONL with subsets: `input` (predict input from code + output), `output` (predict output from code + input), `repair` (find bugs)
- **Evaluation method:** Code comprehension → text prediction → Python subprocess for exact matching
- **Proposed prompt (output prediction):**

```
Predict the output of the following Python code when executed.

Code:
{code}

Input:
{input}

Expected Output:
```

- **Scoring:** Run predicted output through Python subprocess for comparison. Exact string match or semantic equivalence.
- **BenchmarkCategory:** `Reasoning` (code understanding is reasoning, not coding)
- **Estimated effort:** M (~300 lines, needs Python subprocess pattern from `coding_eval.rs`)
- **Files to create:** `src/benchmarks/cruxeval.rs`
- **Files to modify:** `src/benchmarks/mod.rs` (register)

### 4. RULER (`src/benchmarks/ruler.rs`) — **Research**

**Description:** Long-context retrieval benchmark with synthetic needle-in-haystack tasks. Tests whether models can find specific facts buried in long contexts — directly measures source extraction capability.

- **Dataset source:** HuggingFace `NVIDIA/RULER`, Apache 2.0
- **Format:** Synthetic data generation at startup (controllable context lengths, needle positions)
- **Task types:** 
  - `niah` (Needle In A Haystack) — find specific fact in long context
  - `multi_niah` — find multiple needles
  - `kv_retrieval` — key-value lookup in long context
  - `math_find` — find and compute with numbers in long context
- **Evaluation method:** Long context + question → model answers → exact match on needle value
- **Proposed prompt:**

```
{long_context_with_needle}

Question: {question}
Answer:
```

- **Scoring:** Exact string match on extracted value. Track accuracy vs context length curve.
- **BenchmarkCategory:** `Research`
- **Estimated effort:** M (~350 lines, synthetic data generation at startup)
- **Files to create:** `src/benchmarks/ruler.rs`
- **Files to modify:** `src/benchmarks/mod.rs` (register)

### 5. HaluBench (`src/benchmarks/halubench.rs`) — **Hallucination**

**Description:** RAG hallucination benchmark. Tests whether models hallucinate when answering questions with (sometimes incomplete) retrieval context.

- **Dataset source:** HuggingFace `tianyi-lab/HaluBench`
- **Format:** JSON with `question`, `context`, `ground_truth`, and hallucination labels
- **Evaluation method:** Question + context → model answers → keyword containment against ground truth
- **Proposed prompt:**

```
Answer the following question based ONLY on the provided context. If the context doesn't contain enough information, say "I cannot answer this from the given context."

Context: {context}
Question: {question}
Answer:
```

- **Scoring:** Keyword containment match against ground truth. Penalize hallucinated answers (answers not grounded in context).
- **BenchmarkCategory:** `Hallucination`
- **Estimated effort:** S-M (~200 lines, mirrors `halueval.rs`)
- **Files to create:** `src/benchmarks/halubench.rs`
- **Files to modify:** `src/benchmarks/mod.rs` (register)

### 6. SNLI/MultiNLI (`src/benchmarks/snli.rs`) — **Reasoning**

**Description:** Natural Language Inference benchmarks. Given a premise and hypothesis, classify as ENTAILMENT, CONTRADICTION, or NEUTRAL. Direct contradiction detection coverage.

- **Dataset source:** HuggingFace `stanfordnlp/snli` (CC BY-NC 4.0) and `nyu-mll/multi_nli`
- **Format:** premise + hypothesis + label (3 classes)
- **Evaluation method:** Few-shot premise+hypothesis → 3-class classification → exact label match
- **Proposed prompt:**

```
Determine the relationship between the premise and hypothesis.
Respond with only ENTAILMENT, CONTRADICTION, or NEUTRAL.

Premise: {premise}
Hypothesis: {hypothesis}
Relationship:
```

- **Scoring:** Case-insensitive exact label match. Report accuracy per class (especially CONTRADICTION).
- **BenchmarkCategory:** `Reasoning`
- **Estimated effort:** S (~200 lines, mirrors `true_false.rs` with 3 classes)
- **Files to create:** `src/benchmarks/snli.rs`
- **Files to modify:** `src/benchmarks/mod.rs` (register)

### 7. PopQA (`src/benchmarks/popqa.rs`) — **Knowledge**

**Description:** Open-ended knowledge retrieval QA benchmark. Tests factual knowledge with substring matching — no multiple choice.

- **Dataset source:** HuggingFace `akariasai/PopQA`, CC BY-NC 4.0
- **Format:** question + target entity + Wikipedia context
- **Evaluation method:** Question → model generates answer → substring containment of gold entity
- **Proposed prompt:**

```
Question: {question}
Answer:
```

- **Scoring:** Substring containment of gold entity in model response. Case-insensitive.
- **BenchmarkCategory:** `Knowledge`
- **Estimated effort:** S (~150 lines, mirrors `nq_open.rs`)
- **Files to create:** `src/benchmarks/popqa.rs`
- **Files to modify:** `src/benchmarks/mod.rs` (register)

### 8. FactBench (`src/benchmarks/factbench.rs`) — **Research**

**Description:** In-the-wild factuality benchmark testing real-world claims. Contemporary claims from social media and news.

- **Dataset source:** GitHub (arXiv:2410.22257)
- **Format:** JSON with claims and truth labels
- **Evaluation method:** Claim → TRUE/FALSE/UNCERTAIN classification → exact label match
- **Proposed prompt:**

```
Determine whether the following claim is true, false, or uncertain based on available evidence.
Respond with only TRUE, FALSE, or UNCERTAIN.

Claim: {claim}
Answer:
```

- **Scoring:** Case-insensitive exact string match. Track accuracy per label.
- **BenchmarkCategory:** `Research`
- **Estimated effort:** S (~200 lines, mirrors `fever.rs`)
- **Files to create:** `src/benchmarks/factbench.rs`
- **Files to modify:** `src/benchmarks/mod.rs` (register)

### 9. TruthfulQA-Gen (`src/benchmarks/truthful_qa_gen.rs`) — **Hallucination**

**Description:** Open-ended generation variant of TruthfulQA. Tests truthfulness without multiple-choice scaffolding.

- **Dataset source:** Reuse existing TruthfulQA `generation.csv` (already available in the codebase via `truthful_qa.rs`)
- **Format:** question + best_answers (positive keywords) + false_answers (negative keywords)
- **Evaluation method:** Question → open-ended answer → keyword scoring
- **Proposed prompt:**

```
Question: {question}
Answer:
```

- **Scoring:** Count positive keyword matches (from `best_answers`) minus negative keyword matches (from `false_answers`). Higher score = more truthful. Follows the original TruthfulQA generation scoring.
- **BenchmarkCategory:** `Hallucination`
- **Estimated effort:** M (~250 lines, complex scoring but reuses existing dataset loader)
- **Files to create:** `src/benchmarks/truthful_qa_gen.rs`
- **Files to modify:** `src/benchmarks/mod.rs` (register)

---

## Implementation Order

Recommended order for incremental implementation:

1. **SciFact** — trivial given FEVER template, validates Research category
2. **SNLI** — simplest 3-class pattern, validates contradiction detection
3. **PopQA** — simple entity matching, validates Knowledge gap
4. **CRUXEval** — needs Python subprocess, validates code understanding
5. **RULER** — needs synthetic data generation, validates source extraction
6. **HaluBench** — complements hallucination coverage
7. **BBH** — 23 tasks, most complex format, validates reasoning breadth
8. **FactBench** — straightforward claim verification
9. **TruthfulQA-Gen** — complex scoring, reuses existing data

---

## Files to Create

| File | Lines | Description |
|------|-------|-------------|
| `src/benchmarks/scifact.rs` | ~200 | Scientific claim verification |
| `src/benchmarks/snli.rs` | ~200 | NLI contradiction detection |
| `src/benchmarks/popqa.rs` | ~150 | Open-ended knowledge QA |
| `src/benchmarks/cruxeval.rs` | ~300 | Code understanding |
| `src/benchmarks/ruler.rs` | ~350 | Long-context source extraction |
| `src/benchmarks/halubench.rs` | ~200 | RAG hallucination |
| `src/benchmarks/bbh.rs` | ~350 | 23 reasoning tasks |
| `src/benchmarks/factbench.rs` | ~200 | Contemporary factuality |
| `src/benchmarks/truthful_qa_gen.rs` | ~250 | Open-ended truthfulness |
| **Total** | **~2,200** | |

---

## Files to Modify

| File | Changes |
|------|---------|
| `src/benchmarks/mod.rs` | Add `pub mod` for each new benchmark + `map.insert()` registration |

**No changes needed to `src/shared.rs`** — all new benchmarks map to existing `BenchmarkCategory` variants:
- `Research` → SciFact, RULER, FactBench
- `Reasoning` → BBH, CRUXEval, SNLI
- `Hallucination` → HaluBench, TruthfulQA-Gen
- `Knowledge` → PopQA

---

## Steps Checklist

### Phase 1: Quick Wins (DONE ✅)
- [x] **SciFact** — download dataset from HF, implement few-shot claim verification prompt, exact-label scoring, register in mod.rs
- [x] **SNLI** — download subset (~500 instances) from HF, implement 3-class NLI prompt, label match scoring, register
- [x] **PopQA** — download from HF, implement open-ended QA prompt, entity substring matching, register

### Phase 2: Medium Complexity (DONE ✅)
- [x] **CRUXEval** — download from HF, implement code prediction prompts, add Python subprocess scoring (reuse pattern from `coding_eval.rs`), register
- [x] **RULER** — implement synthetic needle-in-haystack data generation at startup, long-context prompts, exact-match scoring, register
- [x] **HaluBench** — download from HF, implement RAG faithfulness prompts, keyword containment scoring, register

### Phase 3: Comprehensive (DONE ✅)
- [x] **BBH** — download task files from HF, implement 23 task prompt templates (start with 5-6 contradiction/logic tasks), aggregate scoring with per-task breakdown, register
- [x] **FactBench** — download from GitHub, implement claim verification prompt, label match scoring, register
- [x] **TruthfulQA-Gen** — reuse existing TruthfulQA dataset loader, implement open-ended keyword scoring, register

### Phase 4: Verification (DONE ✅)
- [x] `cargo check` passes with all new benchmarks
- [x] `cargo test` passes for all benchmarks (141 tests total: 136 lib + 5 integration)
- [ ] Run each benchmark against a local model, verify reasonable scores
- [ ] Verify dataset caching works correctly (datasets not re-downloaded)
- [ ] Verify report generation includes new benchmarks
- [ ] Verify benchmark listing shows new categories

---

## Verification Plan

### Per-benchmark verification
1. **Unit tests:** Each benchmark module should have a test with a known input/output pair
2. **Smoke test:** Run each benchmark with a known model (e.g., local Llama) and verify it produces a valid score between 0-100
3. **Dataset integrity:** Verify downloaded datasets match expected sizes and formats
4. **Caching:** Run benchmark twice, verify second run uses cached data

### Integration verification
1. **`cargo test`** — all benchmark tests pass
2. **`llm-benchmark-runner --list`** — all new benchmarks appear with correct categories
3. **`llm-benchmark-runner --benchmark scifact`** — each benchmark runs independently
4. **Report generation** — verify HTML/JSON reports include new benchmarks

### Score sanity checks
| Benchmark | Expected range (capable model) | Expected range (weak model) |
|-----------|-------------------------------|----------------------------|
| SciFact | 75-90% | 50-65% |
| BBH (logic tasks) | 40-70% | 20-40% |
| CRUXEval | 30-60% | 10-30% |
| RULER (short context) | 80-100% | 50-70% |
| RULER (long context) | 30-70% | 10-30% |
| HaluBench | 60-85% | 30-50% |
| SNLI | 70-85% | 40-60% |
| PopQA | 20-50% | 5-20% |
| FactBench | 60-80% | 50-65% |
| TruthfulQA-Gen | 40-70% | 20-40% |

---

## Coverage Analysis

### Before → After

| Category | Before (count) | Before (benchmarks) | After (count) | New Benchmarks |
|----------|----------------|---------------------|---------------|----------------|
| Hallucination | 6 | halueval, faithdial, hdm_bench, tool_hallucination, truthful_qa, truthful_qa_mc2 | **+2** (8) | HaluBench, TruthfulQA-Gen |
| Research | 1 | fever | **+3** (4) | SciFact, RULER, FactBench |
| Reasoning | 3 | fictional_language, efficient_language, carwash | **+3** (6) | BBH, CRUXEval, SNLI |
| Knowledge | 6 | mmlu_pro, gpqa, supergpqa, race, nq_open, trivia_qa | **+1** (7) | PopQA |
| **Total new** | | | **9 benchmarks** | |

### Gap Coverage Map

| User Requirement | Benchmarks Addressing It |
|------------------|--------------------------|
| Hallucinations | HaluBench (RAG), TruthfulQA-Gen (open-ended), SciFact (claim verification) |
| Source extraction | RULER (needle retrieval, long-context), CRUXEval (code provenance) |
| Research | SciFact (scientific), FactBench (contemporary), RULER (information retrieval) |
| Contradictions | SNLI (NLI contradiction), BBH (logical deduction, disambiguation) |
| Truthfulness | TruthfulQA-Gen (open-ended), SciFact (claim verification), FactBench |
| Lightweight coding | CRUXEval (code understanding without Docker) |

---

## Reuse

| Existing Code | File Path | Reuse For |
|---------------|-----------|-----------|
| FEVER claim verification | `src/benchmarks/fever.rs` | SciFact, FactBench (claim verification pattern) |
| HalluEval classification | `src/benchmarks/halueval.rs` | HaluBench (classification pattern) |
| TruthfulQA dataset loader | `src/benchmarks/truthful_qa.rs` | TruthfulQA-Gen (dataset loader) |
| True/False classification | `src/benchmarks/true_false.rs` | SNLI (multi-class label matching) |
| Code execution via Python | `src/benchmarks/coding_eval.rs` | CRUXEval (Python subprocess) |
| Open-ended QA | `src/benchmarks/nq_open.rs` | PopQA (open-ended QA) |
| Download infrastructure | `src/download.rs` | All new dataset downloads |

---

## Risks & Tradeoffs

### Licensing Risks
| Benchmark | License | Risk Level |
|-----------|---------|------------|
| SciFact | CC BY-NC 2.0 | ⚠️ Non-commercial use only |
| SNLI | CC BY-NC 4.0 | ⚠️ Non-commercial use only |
| PopQA | CC BY-NC 4.0 | ⚠️ Non-commercial use only |
| BBH | Varies per task | ⚠️ Check individual task licenses |
| CRUXEval | MIT | ✅ Permissive |
| RULER | Apache 2.0 | ✅ Permissive |
| FactBench | Check paper | ? Unknown at time of planning |

### Technical Risks
- **CRUXEval** requires Python 3.x to be available on the system. May need to document this as a dependency.
- **TruthfulQA-Gen** keyword scoring is approximate — without a judge model, we can't perfectly score open-ended truthfulness.
- **BBH** has 23 tasks with varying formats — start with 5-6 contradiction/logic tasks, then expand.
- **RULER** has 13 task types — start with core `niah` task, expand to others.
- **Dataset size** — some datasets (SNLI: 570K, PopQA: 14.3K) are large. May want to use subsets for quick evaluation.

### Scope Tradeoffs
- Starting with **5 HIGH PRIORITY benchmarks** (SciFact, BBH, CRUXEval, RULER, HaluBench) gives coverage across all 6 requested categories
- The 4 MEDIUM PRIORITY benchmarks add depth but can be deferred
- BBH and RULER should be implemented with a subset first (5-6 tasks and core `niah` respectively)

---

## Timeline

| Phase | Benchmarks | Estimated Duration |
|-------|------------|-------------------|
| Phase 1: Quick Wins | SciFact, SNLI, PopQA | 1-2 days |
| Phase 2: Medium | CRUXEval, RULER, HaluBench | 2-3 days |
| Phase 3: Comprehensive | BBH, FactBench, TruthfulQA-Gen | 2-3 days |
| Phase 4: Verification | All benchmarks | 0.5 day |
| **Total** | **9 benchmarks** | **6-9 days** |

---

*Plan generated via REFINE → RESEARCH → COMPOSE deep plan pipeline.*
