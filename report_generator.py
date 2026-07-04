#!/usr/bin/env python3
"""
Report generator for benchmark results.
Produces both Markdown and HTML reports.
"""

import os
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, Any

from jinja2 import Template


# HTML template
HTML_TEMPLATE = """<!DOCTYPE html>
<html>
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Model Benchmark Report</title>
  <style>
    body { font-family: Arial, sans-serif; margin: 20px; background: #f5f5f5; }
    h1 { color: #333; }
    h2 { color: #555; border-bottom: 2px solid #007BFF; }
    table { border-collapse: collapse; width: 100%; margin-bottom: 20px; background: white; }
    th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }
    th { background-color: #4CAF50; color: white; }
    tr:nth-child(even) { background-color: #f2f2f2; }
    .highlight { background-color: #ffffcc; }
    .best { background-color: #c8e6c9; }
    .summary { background: white; padding: 15px; border-radius: 5px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
  </style>
</head>
<body>
  <h1>📊 Model Benchmark Report</h1>
  <p>Generated on: {{ timestamp }}</p>

  <h2>Models Evaluated</h2>
  <p>{{ models_evaluated }}</p>

  {% if results.perplexity %}
  <h2>🔍 Perplexity (lower is better)</h2>
  <table>
    <thead>
      <tr><th>Model</th><th>Perplexity</th><th>Method</th></tr>
    </thead>
    <tbody>
      {% for model, data in results.perplexity.items() %}
      <tr class="{% if data.best %}best{% endif %}">
        <td>{{ model }}</td>
        <td>{{ data.value | round(2) }}</td>
        <td>{{ data.method }}</td>
      </tr>
      {% endfor %}
    </tbody>
  </table>
  {% endif %}

  {% if results.mmlu %}
  <h2>📝 MMLU Accuracy & Speed (higher is better)</h2>
  <table>
    <thead>
      <tr><th>Model</th><th>Accuracy</th><th>Total Q</th><th>Correct</th><th>Time/Q (s)</th><th>Q/s</th><th>Total Time (s)</th></tr>
    </thead>
    <tbody>
      {% for model, data in results.mmlu.items() %}
      <tr class="{% if data.best %}best{% endif %}">
        <td>{{ model }}</td>
        <td>{{ data.value | round(2) }}%</td>
        <td>{{ data.total }}</td>
        <td>{{ data.correct }}</td>
        <td>{{ data.speed.avg_time_per_question_s | round(2) }}</td>
        <td>{{ data.speed.questions_per_second | round(1) }}</td>
        <td>{{ data.speed.total_time_s | round(2) }}</td>
      </tr>
      {% endfor %}
    </tbody>
  </table>
  {% endif %}

  {% if results.kld_comparison %}
  <h2>🔁 KLD (Kullback-Leibler Divergence)</h2>
  <p>Average KL divergence between models (lower = more similar output distributions).</p>

  <h3>Average KLD to All Other Models</h3>
  <table>
    <thead>
      <tr><th>Model</th><th>Avg KLD to Others</th></tr>
    </thead>
    <tbody>
      {% for model_name, data in results.kld_comparison.avg_kld_to_others.items() %}
      <tr class="{% if data.best %}best{% endif %}">
        <td>{{ model_name }}</td>
        <td>{{ data.avg_kld_to_others | round(3) }}</td>
      </tr>
      {% endfor %}
    </tbody>
  </table>

  <h3>Pairwise KLD</h3>
  <table>
    <thead>
      <tr><th>Model A</th><th>Model B</th><th>Average KLD</th><th>Target KLD</th><th>Deviation</th><th>Samples</th></tr>
    </thead>
    <tbody>
      {% for pair_key, data in results.kld_comparison.items() %}
      {% if pair_key != 'avg_kld_to_others' %}
      <tr class="{% if data.deviation_from_target is not none and data.deviation_from_target < 0 %}best{% endif %}">
        <td>{{ data.models[0] }}</td>
        <td>{{ data.models[1] }}</td>
        <td>{{ data.avg_kld | round(3) }}</td>
        <td>{{ data.target_kld | round(3) if data.target_kld else 'N/A' }}</td>
        <td>{{ data.deviation_from_target | round(3) if data.deviation_from_target is not none else 'N/A' }}</td>
        <td>{{ data.num_prompts_evaluated }}</td>
      </tr>
      {% endif %}
      {% endfor %}
    </tbody>
  </table>
  {% endif %}

  <div class="summary">
    <h3>📌 Summary</h3>
    <ul>
      {% for item in summary %}
      <li>{{ item }}</li>
      {% endfor %}
    </ul>
  </div>
</body>
</html>"""


def process_results(results: Dict[str, Any]) -> dict:
    """Transform raw results into a report-friendly format."""
    report_results = {"perplexity": {}, "mmlu": {}, "speed": {}, "kld_comparison": results.get("kld_comparison", {})}

    for model_name, benchmarks in results.items():
        if model_name == "kld_comparison":
            continue
        if "perplexity" in benchmarks and benchmarks["perplexity"].get("perplexity", 0) > 0:
            report_results["perplexity"][model_name] = {
                "value": benchmarks["perplexity"]["perplexity"],
                "method": benchmarks["perplexity"].get("method", "unknown"),
            }
        if "mmlu" in benchmarks and "mmlu_accuracy" in benchmarks["mmlu"]:
            report_results["mmlu"][model_name] = {
                "value": benchmarks["mmlu"]["mmlu_accuracy"] * 100,
                "total": benchmarks["mmlu"]["total_questions"],
                "correct": benchmarks["mmlu"]["correct"],
                "speed": benchmarks["mmlu"].get("mmlu_speed", {}),
            }
        if "speed" in benchmarks and "tokens_per_second" in benchmarks["speed"]:
            report_results["speed"][model_name] = {
                "tokens_per_second": benchmarks["speed"]["tokens_per_second"],
                "total_tokens": benchmarks["speed"].get("total_tokens", 0),
                "total_time_seconds": benchmarks["speed"].get("total_time_seconds", 0),
                "avg_time_per_token_ms": benchmarks["speed"].get("avg_time_per_token_ms", 0),
            }

    # Mark best scores
    for key in ["perplexity", "mmlu"]:
        if report_results[key]:
            if key == "perplexity":
                # lower is better
                best_val = min(d["value"] for d in report_results[key].values())
                for m in report_results[key]:
                    report_results[key][m]["best"] = report_results[key][m]["value"] == best_val
            elif key == "mmlu":
                # higher is better
                best_val = max(d["value"] for d in report_results[key].values())
                for m in report_results[key]:
                    report_results[key][m]["best"] = report_results[key][m]["value"] == best_val

    # Mark best for avg_kld_to_others (lower is better)
    if "avg_kld_to_others" in report_results.get("kld_comparison", {}):
        for model_name, data in report_results["kld_comparison"]["avg_kld_to_others"].items():
            pass  # will be computed below
        kld_other_dict = report_results["kld_comparison"]["avg_kld_to_others"]
        best_kld = min(d["avg_kld_to_others"] for d in kld_other_dict.values())
        for model_name in kld_other_dict:
            kld_other_dict[model_name]["best"] = kld_other_dict[model_name]["avg_kld_to_others"] == best_kld

    return report_results


def generate_summary(report_results: dict) -> list:
    """Generate a human-readable summary."""
    summary = []
    if report_results["perplexity"]:
        best_model = [m for m, d in report_results["perplexity"].items() if d.get("best")][0]
        summary.append(f"Lowest perplexity: {best_model} ({min(d['value'] for d in report_results['perplexity'].values()):.2f})")
    if report_results["mmlu"]:
        best_model = [m for m, d in report_results["mmlu"].items() if d.get("best")][0]
        summary.append(f"Highest MMLU accuracy: {best_model} ({max(d['value'] for d in report_results['mmlu'].values()):.1f}%)")
    if report_results["speed"]:
        best_model = [m for m, d in report_results["speed"].items() if d.get("best")][0]
        summary.append(f"Fastest: {best_model} ({max(d['tokens_per_second'] for d in report_results['speed'].values()):.1f} tokens/s)")
    if report_results["speed"]:
        best_model = [m for m, d in report_results["speed"].items() if d.get("best")][0]
        summary.append(f"Fastest: {best_model} ({max(d['tokens_per_second'] for d in report_results['speed'].values()):.1f} tokens/s)")
    if report_results["kld_comparison"]:
        # Average KLD to others
        kld_other_dict = report_results["kld_comparison"].get("avg_kld_to_others", {})
        if kld_other_dict:
            best_model = [m for m, d in kld_other_dict.items() if d.get("best")][0]
            summary.append(f"Lowest avg KLD to others: {best_model} ({min(d['avg_kld_to_others'] for d in kld_other_dict.values()):.3f})")
        # Pairwise KLD
        for pair_key, data in report_results["kld_comparison"].items():
            if pair_key == "avg_kld_to_others":
                continue
            if data.get("deviation_from_target") is not None:
                summary.append(f"KLD {data['models'][0]} vs {data['models'][1]}: {data['avg_kld']:.3f} (target: {data['target_kld']:.3f}, deviation: {data['deviation_from_target']:+.3f})")
            else:
                summary.append(f"KLD {data['models'][0]} vs {data['models'][1]}: {data['avg_kld']:.3f}")
    return summary


def generate_report(results: Dict[str, Any], output_dir: Path):
    """Generate Markdown and HTML reports."""
    report_results = process_results(results)
    summary = generate_summary(report_results)
    models_evaluated = list(results.keys())

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Markdown report
    md_content = f"""# Model Benchmark Report

**Generated on:** {timestamp}

## Models Evaluated

{', '.join(models_evaluated)}

## Perplexity (lower is better)

| Model | Perplexity | Method |
|-------|-----------|--------|
"""
    if report_results["perplexity"]:
        for model, data in report_results["perplexity"].items():
            md_content += f"| {model} | {data['value']:.2f} | {data['method']} |\n"

    md_content += """\n## MMLU Accuracy (higher is better)

| Model | Accuracy | Total Q | Correct |
|-------|----------|---------|---------|
"""
    if report_results["mmlu"]:
        for model, data in report_results["mmlu"].items():
            md_content += f"| {model} | {data['value']:.1f}% | {data['total']} | {data['correct']} |\n"

    md_content += """\n## Speed (tokens per second, higher is better)

| Model | Tokens/sec | Total Tokens | Total Time (s) | Avg Token Time (ms) |
|-------|------------|--------------|----------------|---------------------|
"""
    if report_results["speed"]:
        for model, data in report_results["speed"].items():
            md_content += f"| {model} | {data['tokens_per_second']:.2f} | {data['total_tokens']} | {data['total_time_seconds']:.2f} | {data['avg_time_per_token_ms']:.1f} |\n"

    md_content += """\n## KLD (Kullback-Leibler Divergence)

Average KL divergence between models (lower = more similar output distributions). Target KLD is a published reference value from a benchmark.

### Average KLD to All Other Models

| Model | Avg KLD to Others |
|-------|-------------------|
"""
    avg_kld_others = report_results["kld_comparison"].get("avg_kld_to_others", {})
    if avg_kld_others:
        for model_name, data in avg_kld_others.items():
            md_content += f"| {model_name} | {data['avg_kld_to_others']:.3f} |\n"

    md_content += """\n### Pairwise KLD

| Model A | Model B | Average KLD | Target KLD | Deviation | Samples |
|---------|---------|-------------|------------|-----------|---------|
"""
    for pair_key, data in report_results["kld_comparison"].items():
        if pair_key == "avg_kld_to_others":
            continue
        target = f"{data['target_kld']:.3f}" if data.get('target_kld') else "N/A"
        deviation = f"{data['deviation_from_target']:+.3f}" if data.get('deviation_from_target') is not None else "N/A"
        md_content += f"| {data['models'][0]} | {data['models'][1]} | {data['avg_kld']:.3f} | {target} | {deviation} | {data['num_prompts_evaluated']} |\n"

    md_content += """\n## Summary

"""
    for item in summary:
        md_content += f"- {item}\n"

    md_path = output_dir / "benchmark_report.md"
    with open(md_path, 'w') as f:
        f.write(md_content)
    print(f"Markdown report: {md_path}")

    # HTML report
    html = Template(HTML_TEMPLATE).render(
        timestamp=timestamp,
        models_evaluated=', '.join(models_evaluated),
        results=report_results,
        summary=summary
    )
    html_path = output_dir / "benchmark_report.html"
    with open(html_path, 'w') as f:
        f.write(html)
    print(f"HTML report: {html_path}")

    # Also save raw results as JSON
    json_path = output_dir / "results.json"
    with open(json_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Raw results: {json_path}")
