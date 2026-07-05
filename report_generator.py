#!/usr/bin/env python3
"""
Report generator for benchmark results.
Produces both Markdown and HTML reports for MMLU-Pro and KLD.
"""

import os
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, Any
from jinja2 import Environment, FileSystemLoader


def process_results(results: Dict[str, Any]) -> dict:
    """Transform raw results into report-friendly format."""
    report = {
        "mmlu_pro": {},
        "kld": results.get("kld_pairwise", {}),
    }

    # Process MMLU-Pro results per model
    for model_name, benchmarks in results.get("models", {}).items():
        if "mmlu_pro" in benchmarks:
            mmlu_data = benchmarks["mmlu_pro"]
            report["mmlu_pro"][model_name] = {
                "accuracy": mmlu_data.get("accuracy", 0) * 100,
                "total_questions": mmlu_data.get("total_questions", 0),
                "results_by_subject": mmlu_data.get("results_by_subject", {}),
            }

    # Mark best MMLU-Pro accuracy
    if report["mmlu_pro"]:
        best_acc = max(d["accuracy"] for d in report["mmlu_pro"].values())
        for model in report["mmlu_pro"]:
            report["mmlu_pro"][model]["best"] = report["mmlu_pro"][model]["accuracy"] == best_acc

    # Mark best avg KLD to others (lower is better)
    if "avg_kld_to_others" in report["kld"]:
        kld_other = report["kld"]["avg_kld_to_others"]
        if kld_other:
            best_kld = min(d["avg_kld_to_others"] for d in kld_other.values())
            for model in kld_other:
                kld_other[model]["best"] = kld_other[model]["avg_kld_to_others"] == best_kld

    return report


def generate_summary(report_results: dict) -> list:
    """Generate human-readable summary."""
    summary = []

    if report_results["mmlu_pro"]:
        best_model = [m for m, d in report_results["mmlu_pro"].items() if d.get("best")][0]
        summary.append(f"Highest MMLU-Pro accuracy: {best_model} ({max(d['accuracy'] for d in report_results['mmlu_pro'].values()):.1f}%)")

    if report_results["kld"] and "avg_kld_to_others" in report_results["kld"]:
        kld_other = report_results["kld"]["avg_kld_to_others"]
        if kld_other:
            best_model = [m for m, d in kld_other.items() if d.get("best")][0]
            summary.append(f"Lowest avg KLD to others: {best_model} ({min(d['avg_kld_to_others'] for d in kld_other.values()):.3f})")
        # Pairwise
        for pair_key, data in report_results["kld"].items():
            if pair_key == "avg_kld_to_others" or not data.get("models"):
                continue
            summary.append(f"KLD {data['models'][0]} vs {data['models'][1]}: {data['avg_kld']:.3f} ({data['num_prompts_evaluated']} prompts)")

    return summary


def generate_report(results: Dict[str, Any], output_dir: Path):
    """Generate Markdown and HTML reports."""
    report = process_results(results)
    summary = generate_summary(report)
    model_names = list(results.get("models", {}).keys())

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Markdown report
    md = f"""# Model Benchmark Report

**Generated on:** {timestamp}

## Models Evaluated

{', '.join(model_names)}

## MMLU-Pro Accuracy (higher is better)

| Model | Overall Accuracy | Total Questions |
|-------|-----------------|-----------------|
"""
    for model, data in report["mmlu_pro"].items():
        md += f"| {model} | {data['accuracy']:.1f}% | {data['total_questions']} |\n"

    # Per-subject breakdown
    md += "\n### Per-Subject Breakdown\n\n| Model | Subject | Accuracy | Correct | Wrong |\n|-------|---------|----------|---------|------|\n"
    for model, data in report["mmlu_pro"].items():
        for subject, sdata in data["results_by_subject"].items():
            acc = sdata.get("acc", 0)
            corr = sdata.get("corr", 0)
            wrong = sdata.get("wrong", 0)
            md += f"| {model} | {subject} | {acc:.1f}% | {corr} | {wrong} |\n"

    # KLD
    md += "\n## KLD (Kullback-Leibler Divergence)\n\nAverage KL divergence (lower = more similar output distributions).\n\n### Average KLD to All Other Models\n\n| Model | Avg KLD |\n|-------|---------|\n"
    avg_kld = report["kld"].get("avg_kld_to_others", {})
    for model, data in avg_kld.items():
        md += f"| {model} | {data['avg_kld_to_others']:.3f} |\n"

    md += "\n### Pairwise KLD\n\n| Model A | Model B | Average KLD | Samples |\n|---------|---------|-------------|---------|\n"
    for pair_key, data in report["kld"].items():
        if pair_key == "avg_kld_to_others" or not data.get("models"):
            continue
        md += f"| {data['models'][0]} | {data['models'][1]} | {data['avg_kld']:.3f} | {data['num_prompts_evaluated']} |\n"

    md += "\n## Summary\n\n"
    for item in summary:
        md += f"- {item}\n"

    md_path = output_dir / "benchmark_report.md"
    with open(md_path, "w") as f:
        f.write(md)
    print(f"Markdown report: {md_path}")

    # HTML report — load template from file
    template_dir = Path(__file__).parent / "templates"
    env = Environment(loader=FileSystemLoader(str(template_dir)))
    template = env.get_template("report.html.j2")
    html = template.render(
        timestamp=timestamp,
        models_evaluated=', '.join(model_names),
        mmlu_pro_results=report["mmlu_pro"],
        kld_results=report["kld"],
        summary=summary,
    )
    html_path = output_dir / "benchmark_report.html"
    with open(html_path, "w") as f:
        f.write(html)
    print(f"HTML report: {html_path}")

    # Save raw results as JSON
    json_path = output_dir / "results.json"
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"Raw results: {json_path}")