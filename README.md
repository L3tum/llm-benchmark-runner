# Model Benchmark Suite

A modular Python benchmark suite for evaluating multiple LLM models/quantizations using **llama-swap** and quick metrics:

- **Perplexity**: via `llama-perplexity` binary or API fallback
- **MMLU**: accuracy **and speed** (time per question, questions/sec) on a quick subset of MMLU questions
- **KLD**: Kullback-Leibler Divergence between model output distributions on shared prompts, including:
  - **Pairwise KLD** between every model pair
  - **Average KLD to all other models** — a single metric per model showing how close/far it is from the group
  - Optionally supports cached logits for a fixed reference model (e.g., an unquantized base)

## Prerequisites

- **Python 3.10+** with `pip`
- **llama-swap** running (with your models registered), exposing an OpenAI-compatible API (default: `http://localhost:28287/v1`)
- Optionally, `llama-perplexity` binary (part of [llama.cpp](https://github.com/ggml-org/llama.cpp)) for more accurate perplexity

## Installation

```bash
cd model_benchmark_suite
pip install -r requirements.txt
```

## Configuration

Edit `models_config.yaml` to declare your models. Each model needs:

- `display_name`: human-readable name
- `llama_swap_model`: the model name as registered in llama-swap (used for API calls)
- `gguf_path` **(optional)**: path to the GGUF file for running the perplexity binary. If omitted or set to `null`, perplexity will either fall back to the API or be skipped. You can explicitly skip perplexity with `skip_perplexity: true`.

Example:
```yaml
models:
  - display_name: "MyModel Q4_K_M"
    llama_swap_model: "mymodel-q4"
    gguf_path: "/home/user/models/mymodel.Q4_K_M.gguf"
  - display_name: "MyModel Q5_K_M"
    llama_swap_model: "mymodel-q5"
    gguf_path: "/home/user/models/mymodel.Q5_K_M.gguf"
  - display_name: "MyModel Q8_0"
    llama_swap_model: "mymodel-q8"
    gguf_path: "/home/user/models/mymodel.Q8_0.gguf"

# Global sample size for MMLU and KLD prompts
num_samples: 100     # MMLU questions AND KLD prompts (if source=mmlu)

api_url: "http://localhost:28287/v1"
llama_perplexity_path: "llama-perplexity"

mmlu:
  num_samples: null  # uses global if null
  subset_path: null

kld:
  num_prompts: null
  prompt_source: "mmlu"
  custom_prompts_path: null
```

### Optional: Cached Reference Models

You can include a model that uses **pre-computed logits** (from an unquantized base or any other model), without needing to run it again. This is optional — you can simply compare your quantized models against each other directly.

If you want a fixed reference point (e.g., the unquantized base), add a cached model:

```yaml
  - display_name: "MyModel Base (cached)"
    llama_swap_model: "dummy"
    cached_logits_path: "/home/user/models/base_logits.json"
    gguf_path: null
```

The JSON file should contain a list of objects, each with `prompt` and `top_logprobs` (list of `{token, logprob}` entries). See `data/mymodel_base_toplogprobs.json` for an example.

### Comparing Against a Published KLD (Target KLD)

If you have a known KLD value from a published benchmark (e.g., from a paper or leaderboard), you can set a **target KLD** for each model. The suite will then report how far your actual KLD deviates from the published value — without needing to run the base model at all.

Add `target_kld` to your model configuration:
```yaml
  - display_name: "MyModel Q4"
    llama_swap_model: "mymodel-q4"
    gguf_path: "/path/to/model.Q4.gguf"
    target_kld: 0.12   # published KLD value for this quantization
```

For this to work, you also need a **cached reference model** (the unquantized base) with its logits file. The actual KLD between the model and the reference will be compared to `target_kld`, and the report shows the deviation. If no cached reference is available, the target is shown as a reference but no deviation is computed.

## Running Benchmarks

Start llama-swap with your models, then run:

```bash
python benchmark_runner.py models_config.yaml
```

The script will:

1. Run perplexity for each model (tries `llama-perplexity` binary first, falls back to API)
2. Run MMLU quick test — measures **accuracy and speed** (time per question, questions/sec)
3. Compute **pairwise KLD** between all model pairs, plus **average KLD to all other models** for each model

All results are saved in `benchmark_results/`:

- `benchmark_report.md` – Markdown report
- `benchmark_report.html` – Interactive HTML report with styled tables
- `results.json` – Raw benchmark data for further analysis

## Adding New Benchmarks

The suite is designed to be extensible. To add a new benchmark task:

1. Create a new Python module in `benchmarks/` (e.g., `my_bench.py`)
2. Implement a function `run_my_bench(model: dict, api_url: str, config: dict) -> dict`
3. Add it to `benchmark_runner.py`'s `run_benchmark` function
4. Update the report generator to handle the new result key

## License

MIT