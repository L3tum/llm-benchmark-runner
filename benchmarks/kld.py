#!/usr/bin/env python3
"""
KLD (Kullback-Leibler Divergence) benchmark between two models.
Computes KL divergence between model output distributions on shared prompts.
"""

import json
import os
from typing import List, Dict, Any, Tuple
from openai import OpenAI
import numpy as np
from tqdm import tqdm
from benchmarks.mmlu import fetch_mmlu_subset


def compute_kl_divergence(model_a_logits: dict, model_b_logits: dict) -> float:
    """
    Compute KL divergence between two probability distributions.
    Input: two dicts mapping token indices to logprob (or prob).
    Output: KL(p || q) in nats.
    """
    # Ensure both have the same tokens
    common_tokens = set(model_a_logits.keys()) & set(model_b_logits.keys())
    if not common_tokens:
        return float('inf')

    kl = 0.0
    for token in common_tokens:
        p = model_a_logits[token]
        q = model_b_logits[token]
        if p > 0 and q > 0:
            kl += p * np.log(p / q)
        elif p > 0 and q == 0:
            kl = float('inf')
    return kl


def compute_kl_from_logprobs(logprobs_a: list, logprobs_b: list) -> float:
    """
    Compute KL divergence between two models' next-token distributions.
    Each logprobs is a list of dicts: [{"token": t, "logprob": l}, ...]
    """
    # Convert to probability distributions over common tokens
    def logprobs_to_dist(lp):
        dist = {}
        max_logprob = max(x['logprob'] for x in lp) if lp else 0
        for entry in lp:
            token = entry['token']
            logprob = entry['logprob']
            # Softmax normalization
            dist[token] = np.exp(logprob - max_logprob)
        # Normalize
        total = sum(dist.values())
        for token in dist:
            dist[token] /= total
        return dist

    dist_a = logprobs_to_dist(logprobs_a)
    dist_b = logprobs_to_dist(logprobs_b)

    # Compute KL
    kl = 0.0
    for token, p in dist_a.items():
        if token in dist_b and dist_b[token] > 0:
            q = dist_b[token]
            if p > 0:
                kl += p * np.log(p / q)
        else:
            if p > 0:
                kl = float('inf')
    return kl


def run_kld(models: List[dict], api_url: str, kld_config: dict, mmlu_config: dict = None) -> Dict[str, Any]:
    """
    Run KLD pairwise comparison for all model pairs.
    Supports models with cached logits (pre-computed reference distributions).
    """
    num_prompts = kld_config.get("num_prompts", 30)
    prompt_source = kld_config.get("prompt_source", "mmlu")
    custom_prompts_path = kld_config.get("custom_prompts_path")

    # Get prompts
    prompts = []
    if prompt_source == "mmlu":
        questions = fetch_mmlu_subset(api_url, num_prompts, mmlu_config.get("subset_path") if mmlu_config else None)
        prompts = [q['question'] for q in questions]
    elif prompt_source == "custom" and custom_prompts_path:
        if os.path.exists(custom_prompts_path):
            with open(custom_prompts_path, 'r') as f:
                prompts = [line.strip() for line in f if line.strip()]
        else:
            return {"error": "Custom prompts file not found"}
    else:
        return {"error": "Invalid prompt_source configuration"}

    prompts = prompts[:num_prompts]
    if not prompts:
        return {"error": "No prompts available"}

    client = OpenAI(base_url=api_url, api_key="not-needed")

    # Precompute model distributions for each prompt
    model_dists = {}
    for model in models:
        model_name = model.get("display_name")  # use display_name for clarity
        cached_path = model.get("cached_logits_path")
        model_dists[model_name] = []

        if cached_path:
            # Load cached logits from JSON file
            print(f"Loading cached logits for {model_name} from {cached_path}")
            if not os.path.exists(cached_path):
                print(f"Cached logits file not found: {cached_path} — skipping {model_name}")
                continue
            try:
                with open(cached_path, 'r') as f:
                    cached_data = json.load(f)
                # The file contains a list of top_logprobs dicts per prompt
                # Structure: [{"prompt": "...", "top_logprobs": [...]}, ...]
                for item in cached_data:
                    model_dists[model_name].append(item["top_logprobs"])
            except Exception as e:
                print(f"Error loading cached logits for {model_name}: {e}")
        else:
            # Query the model via API
            llama_swap_model = model.get("llama_swap_model")
            if not llama_swap_model:
                print(f"Skipping {model_name}: no llama_swap_model or cached_logits_path")
                continue
            for prompt in tqdm(prompts, desc=f"Getting logits for {model_name}"):
                try:
                    response = client.chat.completions.create(
                        model=llama_swap_model,
                        messages=[{"role": "user", "content": prompt}],
                        temperature=0.0,
                        max_tokens=1,
                        logprobs=True,
                        top_logprobs=10,  # Top 10 to have enough tokens for KL
                    )
                    if response.choices and response.choices[0].logprobs:
                        top_logprobs = response.choices[0].logprobs.content[0].top_logprobs if response.choices[0].logprobs.content else []
                        model_dists[model_name].append(top_logprobs)
                    else:
                        model_dists[model_name].append([])
                except Exception as e:
                    print(f"Error for {model_name} on prompt: {e}")
                    model_dists[model_name].append([])

    # Pairwise KLD
    results = {}
    # Build a mapping from display_name to model metadata
    model_meta = {}
    for model in models:
        display_name = model.get("display_name")
        model_meta[display_name] = model

    model_names = [m.get("display_name") for m in models]
    pairwise_klds = {}  # (a, b) -> avg_kld

    for i, a_name in enumerate(model_names):
        for b_name in model_names[i+1:]:
            kld_values = []
            for idx, (logprobs_a, logprobs_b) in enumerate(zip(model_dists[a_name], model_dists[b_name])):
                if logprobs_a and logprobs_b:
                    kl = compute_kl_from_logprobs(logprobs_a, logprobs_b)
                    if kl != float('inf'):
                        kld_values.append(kl)
            if kld_values:
                avg_kld = np.mean(kld_values)
                pair_key = f"{a_name}_vs_{b_name}"
                results[pair_key] = {
                    "models": [a_name, b_name],
                    "avg_kld": avg_kld,
                    "num_prompts_evaluated": len(kld_values),
                    "kld_values": kld_values
                }
                pairwise_klds[(a_name, b_name)] = avg_kld
                pairwise_klds[(b_name, a_name)] = avg_kld

    # Compute each model's average KLD to all other models
    results["avg_kld_to_others"] = {}
    for model_name in model_names:
        kld_to_others = [pairwise_klds[(model_name, other)] for other in model_names if other != model_name]
        if kld_to_others:
            avg = np.mean(kld_to_others)
            results["avg_kld_to_others"][model_name] = {
                "avg_kld_to_others": avg,
                "klds": kld_to_others
            }

    # Compare against target KLD (if set) using the cached reference model
    for model in models:
        target_kld = model.get("target_kld")
        if target_kld is None:
            continue
        display_name = model.get("display_name")
        # Find the cached reference model(s)
        for other_model in models:
            other_display = other_model.get("display_name")
            if other_model.get("cached_logits_path"):
                pair_key = f"{display_name}_vs_{other_display}"
                if pair_key in results:
                    actual_kld = results[pair_key]["avg_kld"]
                    deviation = actual_kld - target_kld
                    results[pair_key]["target_kld"] = target_kld
                    results[pair_key]["actual_kld"] = actual_kld
                    results[pair_key]["deviation_from_target"] = deviation
                    print(f"KLD for {display_name} vs cached reference: {actual_kld:.4f} (target: {target_kld}, deviation: {deviation:+.4f})")

    return results
