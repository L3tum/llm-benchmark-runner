#!/usr/bin/env python3
"""
Main benchmark runner.
Runs multiple benchmark tasks across multiple models and generates a report.
"""

import os
import sys
import yaml
from pathlib import Path
from typing import Dict, Any

from benchmarks.perplexity import run_perplexity
from benchmarks.mmlu import run_mmlu
from benchmarks.kld import run_kld
from report_generator import generate_report


def load_config(config_path: str = "models_config.yaml") -> dict:
    """Load model configuration from YAML file."""
    config_path = Path(config_path)
    if not config_path.exists():
        sys.exit(f"Config file not found: {config_path}")
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def run_benchmark(model: dict, task_name: str, config: dict, api_url: str) -> dict:
    """Run a single benchmark task for a model."""
    print(f"\nRunning {task_name} on {model['display_name']}...")
    if task_name == "perplexity":
        return run_perplexity(model, api_url, config.get("llama_perplexity_path", "llama-perplexity"))
    elif task_name == "mmlu":
        return run_mmlu(model, api_url, config.get("mmlu", {}))
    elif task_name == "kld":
        # KLD is a pairwise comparison, handled separately
        return {}
    else:
        raise ValueError(f"Unknown benchmark task: {task_name}")


def run_all_benchmarks(config: dict) -> Dict[str, Any]:
    """Run all benchmark tasks for all models and return results."""
    api_url = config.get("api_url", "http://localhost:28287/v1")
    models = config.get("models", [])
    if not models:
        sys.exit("No models defined in configuration.")

    global_num_samples = config.get("num_samples", 100)
    print("=" * 60)
    print("Model Benchmark Suite")
    print("=" * 60)
    print(f"API URL: {api_url}")
    print(f"Models: {', '.join(m['display_name'] for m in models)}")
    print(f"Global num_samples: {global_num_samples}")
    print()

    # Store results: {model_display_name: {task_name: task_result}}
    results = {}

    for model in models:
        display_name = model['display_name']
        cached = model.get("cached_logits_path")
        skip_perplexity = model.get("skip_perplexity", False)
        results[display_name] = {}

        # Perplexity: skip for cached models or if explicitly disabled
        if not cached and not skip_perplexity:
            try:
                result = run_benchmark(model, "perplexity", config, api_url)
                results[display_name]["perplexity"] = result
            except Exception as e:
                print(f"Error running perplexity on {display_name}: {e}")
                results[display_name]["perplexity"] = {"error": str(e)}

        # MMLU: skip for cached models (can't run via API)
        if not cached:
            try:
                # Pass global num_samples into mmlu config if not overridden
                mmlu_config = config.get("mmlu", {})
                effective_mmlu_config = dict(mmlu_config)
                if effective_mmlu_config.get("num_samples") is None:
                    effective_mmlu_config["num_samples"] = global_num_samples
                result = run_benchmark(model, "mmlu", effective_mmlu_config, api_url)
                results[display_name]["mmlu"] = result
            except Exception as e:
                print(f"Error running MMLU on {display_name}: {e}")
                results[display_name]["mmlu"] = {"error": str(e)}

        # MMLU speed is captured within the MMLU benchmark itself now

    # Run KLD pairwise comparison (needs all models, including cached reference)
    kld_config = config.get("kld", {})
    effective_kld_config = dict(kld_config)
    if effective_kld_config.get("num_prompts") is None:
        effective_kld_config["num_prompts"] = global_num_samples

    mmlu_config = config.get("mmlu", {})
    effective_mmlu_config = dict(mmlu_config)
    if effective_mmlu_config.get("num_samples") is None:
        effective_mmlu_config["num_samples"] = global_num_samples

    # Attach target_kld to each model config (passed as metadata)
    models_with_metadata = []
    for model in models:
        meta_model = dict(model)
        meta_model["target_kld"] = model.get("target_kld")
        models_with_metadata.append(meta_model)

    # KLD uses the same prompt source as MMLU, so we pass the MMLU config for fetching prompts
    kld_results = run_kld(models_with_metadata, api_url, effective_kld_config, mmlu_config=effective_mmlu_config)
    results["kld_comparison"] = kld_results

    return results


def main():
    config_path = sys.argv[1] if len(sys.argv) > 1 else "models_config.yaml"
    config = load_config(config_path)

    results = run_all_benchmarks(config)

    # Generate report
    output_dir = Path("benchmark_results")
    output_dir.mkdir(exist_ok=True)

    generate_report(results, output_dir)
    print(f"\nReport generated in: {output_dir}")


if __name__ == "__main__":
    main()
