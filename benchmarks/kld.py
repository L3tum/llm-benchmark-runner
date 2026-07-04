#!/usr/bin/env python3
"""
KLD (Kullback-Leibler Divergence) benchmark between model pairs.
Computes KL divergence between model output distributions on shared prompts.
"""

import os
import math
from typing import Dict, List, Any
import numpy as np
from tqdm import tqdm
from openai import OpenAI


def collect_logits(model: dict, config: dict) -> List[dict]:
    """
    Collect logits for KLD prompts from a model's proxy.
    Returns list of top_logprobs dicts (one per prompt).
    Tracks failed prompts and warns if failure rate is high.

    Args:
        model: dict with display_name, proxy, ...
        config: benchmark-specific config with num_prompts, prompt_source, etc.
    """
    proxy = model["proxy"]
    client = OpenAI(base_url=proxy, api_key="not-needed")

    num_prompts = config.get("num_prompts", 100)
    prompt_source = config.get("prompt_source", "mmlu")

    prompts = get_kld_prompts(num_prompts, prompt_source, config)
    if not prompts:
        return []

    logits = []
    failed = 0
    failure_reasons = []
    for prompt in tqdm(prompts, desc="Collecting KLD logits"):
        try:
            response = client.chat.completions.create(
                model="model",  # llama-server style
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=1,
                logprobs=True,
                top_logprobs=10,
            )
            if response.choices and response.choices[0].logprobs and response.choices[0].logprobs.content:
                top_lp = response.choices[0].logprobs.content[0].top_logprobs or []
                logits.append(top_lp)
            else:
                logits.append([])
                failed += 1
                failure_reasons.append("empty response logprobs")
        except Exception as e:
            print(f"KLD logit error: {e}")
            logits.append([])
            failed += 1
            failure_reasons.append(f"exception: {e}")

    if failed > len(prompts) * 0.3:  # warn if >30% failures
        print(f"WARNING: {failed}/{len(prompts)} KLD prompt failures ({failed/len(prompts)*100:.0f}%)")
        # Show most common failure reason
        if failure_reasons:
            print("  Most common reason: " + max(set(failure_reasons), key=failure_reasons.count))
    return logits


def get_kld_prompts(num_prompts: int, prompt_source: str, config: dict) -> List[str]:
    """Get prompts for KLD from MMLU-Pro or custom source."""
    if prompt_source == "mmlu":
        # Use MMLU-Pro questions as prompts
        dataset = load_mmlu_pro_dataset()
        subjects = config.get("subjects") or list(dataset.keys())
        prompts = []
        for subj in subjects:
            prompts.extend([q["question"] for q in dataset[subj][:num_prompts]])
        return prompts[:num_prompts]
    elif prompt_source == "custom":
        custom_path = config.get("custom_prompts_path")
        if custom_path and os.path.exists(custom_path):
            with open(custom_path, "r") as f:
                return [line.strip() for line in f if line.strip()]
        print(f"Warning: custom prompts path '{custom_path}' not found or not set")
        return []
    return []


# Module-level cache for MMLU-Pro dataset (lazy loaded, no config needed)
_mmlu_pro_dataset_cache = None


def load_mmlu_pro_dataset() -> dict:
    """Load MMLU-Pro test set, grouped by category. Cached after first call."""
    global _mmlu_pro_dataset_cache
    if _mmlu_pro_dataset_cache is None:
        from datasets import load_dataset
        print("Loading MMLU-Pro dataset... (cached for subsequent calls)")
        dataset = load_dataset("TIGER-Lab/MMLU-Pro", split="test")
        res = {}
        for item in dataset:
            opts = [o for o in item["options"] if o != "N/A"]
            item["options"] = opts
            cat = item["category"]
            if cat not in res:
                res[cat] = []
            res[cat].append(item)
        _mmlu_pro_dataset_cache = res
    return _mmlu_pro_dataset_cache


def compute_pairwise_kld(all_logits: Dict[str, List[dict]]) -> Dict[str, Any]:
    """
    Compute pairwise KLD for all model pairs from collected logits.
    all_logits: dict mapping model_name -> list of top_logprobs dicts (one per prompt)
    Returns: dict with pairwise KLD results and average KLD to others.
    """
    model_names = list(all_logits.keys())
    pairwise_klds: Dict[str, float] = {}
    results = {}

    for i, a_name in enumerate(model_names):
        for b_name in model_names[i+1:]:
            logits_a = all_logits[a_name]
            logits_b = all_logits[b_name]
            kld_values = []
            for logprobs_a, logprobs_b in tqdm(zip(logits_a, logits_b),
                                              desc=f"KLD: {a_name} vs {b_name}",
                                              total=min(len(logits_a), len(logits_b))):
                kl = compute_kl_from_logprobs(logprobs_a, logprobs_b)
                if math.isfinite(kl):
                    kld_values.append(kl)
            if kld_values:
                avg_kld = float(np.mean(kld_values))
                pair_key = f"{a_name}_vs_{b_name}"
                results[pair_key] = {
                    "models": [a_name, b_name],
                    "avg_kld": avg_kld,
                    "num_prompts_evaluated": len(kld_values),
                    "kld_values": kld_values
                }
                pairwise_klds[(a_name, b_name)] = avg_kld
                pairwise_klds[(b_name, a_name)] = avg_kld

    # Average KLD to all other models
    results["avg_kld_to_others"] = {}
    for model_name in model_names:
        kld_to_others = [pairwise_klds.get((model_name, other), float('inf'))
                         for other in model_names if other != model_name]
        kld_to_others = [k for k in kld_to_others if k != float('inf')]
        if kld_to_others:
            avg = float(np.mean(kld_to_others))
            results["avg_kld_to_others"][model_name] = {
                "avg_kld_to_others": avg,
                "klds": kld_to_others
            }

    return results


def compute_kl_from_logprobs(logprobs_a: list, logprobs_b: list) -> float:
    """
    Compute KL divergence between two models' next-token distributions.
    Each logprobs is a list of dicts: [{"token": t, "logprob": l}, ...]
    Returns KL(p || q) in nats, or inf if either logprobs list is empty.
    """
    if not logprobs_a or not logprobs_b:
        return float('inf')

    # Convert to probability distributions over common tokens
    def logprobs_to_dist(lp):
        dist = {}
        max_logprob = max(x['logprob'] for x in lp)
        for entry in lp:
            token = entry['token']
            logprob = entry['logprob']
            dist[token] = math.exp(logprob - max_logprob)
        # Normalize
        total = sum(dist.values())
        if total == 0:
            return {}
        for token in dist:
            dist[token] /= total
        return dist

    dist_a = logprobs_to_dist(logprobs_a)
    dist_b = logprobs_to_dist(logprobs_b)

    # Compute KL
    kl = 0.0
    for token, p in dist_a.items():
        q = dist_b.get(token, 0.0)
        if q > 0 and p > 0:
            kl += p * math.log(p / q)
        elif p > 0 and q == 0:
            return float('inf')
    return kl