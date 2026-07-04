"""
Extensible benchmark modules.
New benchmarks can be added by creating a Python file in benchmarks/ and
implementing pre_execute, execute, and post_execute functions.
"""

from benchmarks.mmlu_pro import run_mmlu_pro
from benchmarks.kld import collect_logits, compute_pairwise_kld

# Registry of available benchmarks
# Each benchmark can have pre, execute, post phases.
# pre/post may be None (no-op).


def _kld_post(model_results):
    """Wrapper: extract 'kld' logits from per-model results, then compute pairwise KLD."""
    all_logits = {name: data.get("kld", []) for name, data in model_results.items()}
    return compute_pairwise_kld(all_logits)


BENCHMARKS = {
    "mmlu_pro": {
        "pre": None,
        "execute": run_mmlu_pro,
        "post": None,
    },
    "kld": {
        "pre": None,
        "execute": collect_logits,
        "post": _kld_post,  # receives model_results dict, extracts kld, computes pairwise
    },
    # Add new benchmarks here:
    # "arc": {"pre": ..., "execute": ..., "post": ...},
}


def get_benchmarks():
    """Return dict of benchmark name -> function dict (pre/execute/post)."""
    return BENCHMARKS.copy()


def pre_execute_benchmarks(benchmark_name, config):
    """Run pre-execute phase (e.g., download datasets)."""
    if benchmark_name not in BENCHMARKS:
        raise ValueError(f"Unknown benchmark: {benchmark_name}")
    phase = BENCHMARKS[benchmark_name].get("pre")
    if phase:
        return phase(config)


def execute_benchmark(benchmark_name, model, config):
    """Run a single benchmark's execution phase."""
    if benchmark_name not in BENCHMARKS:
        raise ValueError(f"Unknown benchmark: {benchmark_name}")
    return BENCHMARKS[benchmark_name]["execute"](model, config)


def post_execute_benchmarks(benchmark_name, config, model_results):
    """Run post-execute phase (e.g., pairwise KLD)."""
    if benchmark_name not in BENCHMARKS:
        raise ValueError(f"Unknown benchmark: {benchmark_name}")
    phase = BENCHMARKS[benchmark_name].get("post")
    if phase:
        return phase(model_results)


def run_benchmark(benchmark_name, model, config):
    """Deprecated alias for execute_benchmark."""
    return execute_benchmark(benchmark_name, model, config)