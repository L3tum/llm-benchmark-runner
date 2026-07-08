# Evaluation of Adding New Benchmarks

I've researched each of the benchmarks you mentioned, plus some additional long-horizon context options. Below is an assessment of each, including what they test, how they're used, and whether they make sense to add.

## 1. Carwash Benchmark

**What it is:** A simple viral common-sense reasoning test. The prompt asks: *"I want to wash my car. The car wash is 50 meters away from my current location. Should I walk or drive to the car wash?"* The correct answer is **drive**, because you need to take the car with you. Most models fail, answering "walk."

**Key findings:**
- Tested on 131 models; only a handful passed consistently (some sources cite only 5 out of 53 tested)
- Exposes physical reasoning gaps — models fail to track object references in real-world scenarios
- Not a comprehensive benchmark; it's a single-question gotcha, but it's a useful sanity check for basic reasoning

**Pros:**
- Extremely simple, no infrastructure needed
- Excellent "taste test" for common-sense reasoning
- Good for quick sanity checks

**Cons:**
- Too narrow to stand alone as a benchmark
- Only tests one edge case; doesn't scale well

**Verdict:** Worth adding as a quick, easy sanity-check prompt, but **not as a primary benchmark**. It's a useful addition to your existing benchmarks as a "gotcha" to ensure basic reasoning isn't broken.

## 2. IFEval

**What it is:** Instruction-Following Evaluation, an objective benchmark that tests LLMs on verifiable constraints like *"write in more than 400 words"*, *"mention the keyword AI at least 3 times"*, or *"do not use the word 'the'"*.

**Key features:**
- Published by Google, widely adopted, integrated into `lm-evaluation-harness`
- 65+ models on leaderboards, current average score ~0.8, leader at ~0.95
- Includes structural constraints, formatting rules, and negative constraints
- Multilingual version (M-IFEval) also available

**Pros:**
- Objective, machine-verifiable scoring (no human judgment)
- Measures a critical capability: precise instruction adherence
- Widely used and well-understood

**Cons:**
- Doesn't test deep reasoning or creativity — purely mechanical compliance
- Some models can game it by writing filler text to meet length constraints

**Verdict:** **Highly recommended** to add. It complements reasoning benchmarks by measuring a fundamentally different capability — exact instruction following. Especially useful if you're evaluating models for structured output or format-sensitive tasks.

## 3. HarmBench

**What it is:** A standardized safety evaluation framework for LLMs from the Center for AI Safety. It tests both the model's tendency to generate harmful completions and its robustness against adversarial jailbreak attacks.

**Key features:**
- 510 harmful behaviors categorized by function and semantic domain
- Evaluates multiple attack strategies (direct prompting, jailbreaks, red-teaming)
- Quantitative, reproducible — adopted as a de facto standard for automated red-teaming
- Covers jailbreak robustness, not just base harmfulness

**Pros:**
- Comprehensive safety evaluation, widely recognized
- Tests adversarial robustness, which is critical for production models
- Standardized methodology reduces apples-to-oranges comparisons

**Cons:**
- Requires careful setup (adversarial attack pipelines)
- Results can be sensitive to attack strategies chosen
- Not a "capability" benchmark — it's purely a safety/robustness measure

**Verdict:** **Strongly recommended** if you're evaluating models for production or need to understand safety profiles. HarmBench is the gold standard for adversarial safety benchmarking and is essential for any serious model comparison.

## 4. DeepSWE (Deep-SWE)

**What it is:** A long-horizon, contamination-free benchmark for frontier coding agents. It includes 113 original software engineering tasks across 91 repositories and 5 languages (TypeScript, Go, Python, JavaScript, Rust) with hand-written program-based verifiers.

**Key features:**
- Tasks are drawn from active open-source repos, ensuring real-world relevance
- Program-based verifiers test actual software behavior, not just patch correctness
- Designed specifically to separate leading coding agents (e.g., Claude vs. GPT)
- Long-horizon means agents must plan, write, and test code across many steps

**Pros:**
- Measures agentic coding capability — not just code generation, but real software engineering
- High-difficulty, contamination-free tasks
- Program-based verification is objective and robust

**Cons:**
- Requires agentic execution environments (e.g., mini-swe-agent)
- Computationally expensive (long-running tasks)
- Not suitable for non-agentic models (you'd need an agent layer)

**Verdict:** **Recommended** if you're evaluating models as coding agents or if your use case involves agentic software engineering workflows. DeepSWE is the frontier benchmark for this. If you're just evaluating LLMs for code completion, consider SWE-bench Lite instead.

## 5. Long-Horizon Context Tasks

You mentioned "long-horizon context tasks" — I've interpreted this as benchmarks that test LLMs on tasks requiring **long context windows** and/or **extended multi-step reasoning**. Here are the best options:

### A. HELMET (Princeton)

**What it is:** HELMET (How to Evaluate Long-context Language Models Effectively and Thoroughly) is a comprehensive long-context benchmark from Princeton covering seven downstream task categories: RAG, in-context learning, summarization, re-ranking, instruction following, multilingual, and code.

**Key findings:**
- Tests models at 128K+ context lengths
- Demonstrates that needle-in-a-haystack (NIAH) tests don't predict downstream performance
- Application-centric and covers 34GB of data across 7 task types

**Pros:**
- Holistic evaluation of long-context abilities
- Shows what really matters for downstream applications
- Covers diverse tasks, not just retrieval

**Verdict:** **Highly recommended** as the long-context benchmark of choice. It's the most thorough and practical evaluation for long-context models.

### B. LongBench v2 / LongBench Pro

**What it is:** LongBench v2 tests LLMs on long-context tasks with 8K to 2M word inputs, across real-world tasks. LongBench Pro adds bilingual (English/Chinese) and more realistic scenarios.

**Pros:**
- Very long context (up to 2M words in v2)
- Real-world multitasks, not synthetic
- Bilingual coverage in Pro

**Cons:**
- Some tasks overlap with HELMET's coverage

**Verdict:** Good for pushing context length limits beyond what HELMET covers. Add it alongside HELMET for ultra-long context evaluation.

### C. RULER

**What it is:** A benchmark specifically for testing retrieval-augmented generation (RAG) and multi-hop reasoning over long contexts.

**Pros:**
- Focuses on reasoning across long documents
- Designed to complement NIAH

**Cons:**
- Narrower scope than HELMET

**Verdict:** Use if you need a specialized RAG/long-reasoning benchmark; otherwise, HELMET covers this.

### D. DeepSWE (already above)

DeepSWE itself is also a "long-horizon" benchmark in the sense that agents must work through extended multi-step coding tasks — another form of long-horizon reasoning.

## Summary and Recommendations

Based on your goals, here's my recommended addition set:

| Benchmark | Type | Cost/Complexity | Priority | Why |
|-----------|------|-----------------|----------|-----|
| **IFEval** | Instruction following | Low | **High** | Easy to run, critical capability, widely used |
| **HarmBench** | Safety/robustness | High | **High** | Essential for production model evaluation |
| **HELMET** | Long-context downstream | Medium | **High** | Best all-around long-context evaluation |
| **Carwash** | Common-sense reasoning | Negligible | **Medium** | Fun sanity check, but not a serious benchmark |
| **DeepSWE** | Agentic coding | High | **Conditional** | Only if evaluating models as coding agents |
| **LongBench v2** | Ultra-long context | Medium | **Medium** | Good for testing 1M+ token capabilities |

**Overall recommendation:** Add **IFEval** and **HarmBench** as core additions. Add **HELMET** if long-context evaluation is important. Add **Carwash** as a quick sanity-check prompt. DeepSWE and LongBench are valuable for specialized evaluation of coding agents and ultra-long context respectively.
