#!/usr/bin/env python3
"""
Benchmark runner — starts each model, benchmarks it, stops it.
Resumable: results saved after each model, resume skips completed models.
Configurable via --config argument.
Extensible: new benchmarks registered in benchmarks/__init__.py with
pre/execute/post lifecycle.

# ponytail: per-model benchmark selection removed; config["benchmarks"] is the
# single source of truth. If per-model selection is needed, add it.
"""

import os
import sys
import json
import time
import subprocess
import signal
import argparse
import requests
import yaml
from pathlib import Path
from typing import Dict, Any
from tqdm import tqdm

from benchmarks import pre_execute_benchmarks, execute_benchmark, post_execute_benchmarks, get_benchmarks
from report_generator import generate_report


DEFAULT_CONFIG = "models_config.yaml"
RESULTS_FILE = "benchmark_results/results.json"


def load_config(path: str) -> dict:
    """Load YAML config, exit if not found."""
    if not os.path.exists(path):
        sys.exit(f"Config not found: {path}")
    with open(path, "r") as f:
        return yaml.safe_load(f)


def save_results(results: dict, path: str = RESULTS_FILE):
    """Save results atomically, creating output dir if needed."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path + ".tmp"
    with open(tmp_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    os.replace(tmp_path, path)


def load_existing_results(path: str = RESULTS_FILE) -> dict:
    """Load results from previous run if it exists."""
    if os.path.exists(path):
        with open(path, "r") as f:
            return json.load(f)
    return {}


def wait_for_proxy(proxy: str, timeout: int = 60, poll_interval: int = 2):
    """Poll proxy /v1/models endpoint until it responds or times out."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            resp = requests.get(f"{proxy}/models", timeout=5)
            if resp.status_code == 200:
                return True
        except requests.RequestException:
            pass
        time.sleep(poll_interval)
    return False


def start_model(cmd: str) -> subprocess.Popen:
    """Start model via subprocess.Popen (shell=True, new session).
    Uses preexec_fn=os.setsid on Linux to create a process group.
    """
    try:
        # Linux
        return subprocess.Popen(cmd, shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                                preexec_fn=os.setsid)
    except AttributeError:
        # Non-Linux (e.g., Windows), use start_new_session if available
        if hasattr(os, 'setsid'):
            return subprocess.Popen(cmd, shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                                    start_new_session=True)
        else:
            return subprocess.Popen(cmd, shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def stop_model(cmd_stop: str, proc: subprocess.Popen):
    """Stop model: run cmdStop if provided, else SIGTERM.
    If cmdStop is not set, sends SIGTERM to the process group, then SIGKILL fallback.
    """
    stopped_cleanly = True
    if cmd_stop:
        result = subprocess.run(cmd_stop, shell=True, capture_output=True)
        if result.returncode != 0:
            print(f"Warning: cmdStop failed with code {result.returncode}")
            stopped_cleanly = False
        time.sleep(1)  # wait for cleanup
    else:
        try:
            # Kill process group (created by setsid)
            os.killpg(proc.pid, signal.SIGTERM)  # proc.pid is the leader of the group
            proc.wait(timeout=10)
        except (subprocess.TimeoutExpired, ProcessLookupError, OSError):
            stopped_cleanly = False

    # Fallback to SIGKILL if process still running
    if not stopped_cleanly:
        try:
            os.kill(proc.pid, 0)  # check if process still alive
            print("Process still alive, forcing SIGKILL")
            os.killpg(proc.pid, signal.SIGKILL)
            proc.wait(timeout=5)
        except (ProcessLookupError, OSError):
            pass  # process is dead
        except subprocess.TimeoutExpired:
            print("Warning: SIGKILL didn't terminate process within 5s")


def main():
    parser = argparse.ArgumentParser(description="Run benchmark suite on multiple models")
    parser.add_argument("--config", default=DEFAULT_CONFIG,
                        help=f"Path to config file (default: {DEFAULT_CONFIG})")
    parser.add_argument("--no-resume", action="store_true",
                        help="Force rerun all models, ignoring existing results")
    args = parser.parse_args()

    # Load config
    config = load_config(args.config)
    models = config.get("models", [])
    if not models:
        sys.exit("No models defined in config.")

    # Load existing results for resumability
    existing = load_existing_results(RESULTS_FILE) if not args.no_resume else {}
    completed_models = set()
    per_model_benchmarks = {}  # per-model completed benchmark names
    if "models" in existing and isinstance(existing["models"], dict):
        for m_name, m_data in existing["models"].items():
            if isinstance(m_data, dict):
                if "benchmarks_completed" in m_data:
                    per_model_benchmarks[m_name] = set(m_data.get("benchmarks_completed", []))
                    if m_data.get("status") == "completed":
                        completed_models.add(m_name)
                elif m_data.get("status") == "completed":
                    completed_models.add(m_name)

    # Benchmark selection: if config has "benchmarks" list, use it. If empty, run all registered.
    benchmarks = config.get("benchmarks", [])
    if not benchmarks:
        benchmarks = list(get_benchmarks().keys())
        print(f"No benchmarks specified; running all registered: {', '.join(benchmarks)}")
    else:
        print(f"Benchmarks to run: {', '.join(benchmarks)}")

    # Benchmarks config section: per-benchmark settings
    benchmark_config = config.get("benchmark", {})

    print("=" * 60)
    print("Model Benchmark Suite — Direct Execution")
    print("=" * 60)
    print(f"Config: {args.config}")
    print(f"Models: {', '.join(m['display_name'] for m in models)}")
    if args.no_resume:
        print("No-resume mode: ignoring existing results")
    elif completed_models:
        print(f"Resuming — skipping completed: {', '.join(completed_models)}")
    print()

    # All models results accumulator: model_name -> {bench_name: result}
    all_models_results = {}

    # Pre-phase: once before any models run
    print("Pre-execution phase:")
    for bench_name in benchmarks:
        bench_cfg = benchmark_config.get(bench_name, {})
        try:
            pre_execute_benchmarks(bench_name, bench_cfg)
        except Exception as e:
            print(f"Warning: pre-execute for {bench_name} failed: {e}")
    print()

    # Model loop
    for model in models:
        name = model["display_name"]
        if name in completed_models:
            print(f"\nSkipping already completed model: {name}")
            all_models_results[name] = existing["models"][name]  # restore old results
            continue

        print(f"\n{'=' * 40}")
        print(f"Starting model: {name}")
        print(f"Command: {model['cmd']}")
        print(f"Proxy: {model['proxy']}")
        print(f"{'=' * 40}")

        # Start model
        proc = start_model(model["cmd"])
        cmd_stop = model.get("cmdStop")

        # Wait for proxy health
        if not wait_for_proxy(model["proxy"], timeout=60):
            print(f"ERROR: Model {name} proxy did not become healthy within 60s")
            stop_model(cmd_stop, proc)
            continue

        # Run benchmarks, track per-benchmark completion
        model_results = {}
        benchmark_completed = set()

        for bench_name in benchmarks:
            bench_cfg = benchmark_config.get(bench_name, {})
            # Check resume: skip already completed benchmarks for this model
            if name in per_model_benchmarks and bench_name in per_model_benchmarks[name]:
                print(f"Skipping already completed {bench_name} for {name}")
                if "models" in existing and name in existing["models"]:
                    old_model_data = existing["models"][name]
                    if isinstance(old_model_data, dict) and bench_name in old_model_data:
                        model_results[bench_name] = old_model_data[bench_name]
                        benchmark_completed.add(bench_name)
                continue
            try:
                result = execute_benchmark(bench_name, model, bench_cfg)
                model_results[bench_name] = result
                benchmark_completed.add(bench_name)
                print(f"{bench_name} result: {result}")
            except Exception as e:
                print(f"ERROR running {bench_name} for {name}: {e}")
                model_results[bench_name] = {"error": str(e)}
                benchmark_completed.add(bench_name)  # count as completed (failed)

        # Stop model
        stop_model(cmd_stop, proc)
        print(f"Model {name} stopped.")

        # Determine status
        status = "completed" if len(benchmark_completed) > 0 else "error"

        # Build per-model results with status and benchmark tracking
        model_data = {
            "status": status,
            "benchmarks_completed": list(benchmark_completed),
            **model_results,
        }
        all_models_results[name] = model_data

        # Save intermediate results immediately
        save_results({
            "models": all_models_results,
        })

    # Post-phase: once after all models run (e.g., pairwise KLD)
    print("\nPost-execution phase:")
    kld_pairwise = {}
    for bench_name in benchmarks:
        bench_cfg = benchmark_config.get(bench_name, {})
        try:
            result = post_execute_benchmarks(bench_name, bench_cfg, all_models_results)
            # Capture KLD pairwise results if present
            if bench_name == "kld" and result:
                kld_pairwise = result
        except Exception as e:
            print(f"Warning: post-execute for {bench_name} failed: {e}")

    # Build final results dict with pairwise KLD
    final_results = {"models": all_models_results}
    if kld_pairwise:
        final_results["kld_pairwise"] = kld_pairwise

    # Save final results
    save_results(final_results, RESULTS_FILE)

    # Generate reports (Markdown and HTML)
    output_dir = Path("benchmark_results")
    output_dir.mkdir(parents=True, exist_ok=True)
    generate_report(final_results, output_dir)

    print(f"\nBenchmark complete. Results saved to {RESULTS_FILE}")


if __name__ == "__main__":
    main()