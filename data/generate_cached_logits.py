#!/usr/bin/env python3
"""
Generate a cached logits JSON file for a base model.
This script fetches MMLU prompts, runs them through a model, and saves the
top logprobs for each prompt. The output can then be used as a cached reference
in the benchmark suite for KLD comparison without re-running the base model.
"""

import argparse
import json
import os
import sys
from openai import OpenAI
from tqdm import tqdm

# Import the MMLU subset fetcher from the benchmark suite
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from benchmarks.mmlu import fetch_mmlu_subset


def main():
    parser = argparse.ArgumentParser(description="Generate cached logits for a base model.")
    parser.add_argument("--model", required=True,
                        help="Model name as registered in llama-swap (used to query API).")
    parser.add_argument("--api-url", default="http://localhost:28287/v1",
                        help="API URL (default: http://localhost:28287/v1).")
    parser.add_argument("--num-prompts", type=int, default=100,
                        help="Number of prompts to process (default: 100).")
    parser.add_argument("--output", required=True,
                        help="Output JSON file path (e.g., my_base_logits.json).")
    parser.add_argument("--prompt-source", default="mmlu",
                        choices=["mmlu", "custom"],
                        help="Prompt source: 'mmlu' or 'custom' (requires --custom-prompts-path).")
    parser.add_argument("--custom-prompts-path", default=None,
                        help="Path to custom prompts file (JSON/JSONL with 'prompt' field).")
    parser.add_argument("--subset-path", default=None,
                        help="Path to a local MMLU subset file (if prompt-source=mmlu).")
    args = parser.parse_args()

    # Get prompts
    prompts = []
    if args.prompt_source == "mmlu":
        questions = fetch_mmlu_subset(args.api_url, args.num_prompts, args.subset_path)
        prompts = [q['question'] for q in questions[:args.num_prompts]]
    elif args.prompt_source == "custom":
        if not args.custom_prompts_path:
            print("Error: --prompt-source custom requires --custom-prompts-path")
            sys.exit(1)
        if not os.path.exists(args.custom_prompts_path):
            print(f"Error: Custom prompts file not found: {args.custom_prompts_path}")
            sys.exit(1)
        # Load custom prompts (JSON list of prompts, or JSON/JSONL with 'prompt' key)
        with open(args.custom_prompts_path, 'r') as f:
            data = json.load(f) if not args.custom_prompts_path.endswith('.jsonl') else [json.loads(line) for line in f if line.strip()]
        if isinstance(data, list):
            for item in data:
                prompt = item.get("prompt", item) if isinstance(item, dict) else item
                prompts.append(str(prompt))
        else:
            # Assume it's a JSON file with a 'prompts' key
            prompts = data.get("prompts", data)
        prompts = prompts[:args.num_prompts]

    if not prompts:
        print("Error: No prompts available.")
        sys.exit(1)

    print(f"Processing {len(prompts)} prompts for model: {args.model}")

    # Query the model
    client = OpenAI(base_url=args.api_url, api_key="not-needed")
    results = []

    for prompt in tqdm(prompts, desc=f"Querying {args.model}"):
        try:
            response = client.chat.completions.create(
                model=args.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=1,
                logprobs=True,
                top_logprobs=10,  # Top 10 tokens
            )
            top_logprobs = []
            if response.choices and response.choices[0].logprobs and response.choices[0].logprobs.content:
                # The API returns a list of top_logprobs per token
                # We take the first token's top_logprobs
                for entry in response.choices[0].logprobs.content[0].top_logprobs:
                    top_logprobs.append({"token": entry.token, "logprob": entry.logprob})
            results.append({
                "prompt": prompt,
                "top_logprobs": top_logprobs
            })
        except Exception as e:
            print(f"Error on prompt: {e}")
            results.append({"prompt": prompt, "top_logprobs": []})

    # Save the results
    output_dir = os.path.dirname(args.output)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    with open(args.output, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"Saved {len(results)} prompts to {args.output}")
    # Check for empty top_logprobs (API might not return them properly)
    empty_count = sum(1 for r in results if not r['top_logprobs'])
    if empty_count > 0:
        print(f"Warning: {empty_count} prompts had empty top_logprobs — check API response.")


if __name__ == "__main__":
    main()
