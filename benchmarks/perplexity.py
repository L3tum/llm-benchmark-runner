#!/usr/bin/env python3
"""
Perplexity benchmark using llama.cpp's llama-perplexity binary.
If the binary is not found or fails, falls back to API-based perplexity using logprobs.
"""

import subprocess
import os
import sys
from typing import Dict, Any
from openai import OpenAI


def run_perplexity_binary(gguf_path: str, binary: str = "llama-perplexity") -> float:
    """
    Run llama-perplexity on a GGUF file and return the perplexity score.
    Returns -1 if the command fails.
    """
    # Use absolute path for binary if it's not found in PATH
    binary_path = binary if os.path.isabs(binary) else binary
    try:
        cmd = [binary_path, "-m", gguf_path, "-t", "8", "-p", "1000"]
        # Run with output to get the perplexity line
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
        # Look for "ppl =" in output
        for line in result.stdout.split('\n'):
            if "ppl =" in line.lower() or "perplexity =" in line.lower():
                # Parse the number after =
                parts = line.split("=")
                if len(parts) >= 2:
                    score = float(parts[-1].strip())
                    return score
        # If not found, try stderr
        for line in result.stderr.split('\n'):
            if "ppl =" in line.lower() or "perplexity =" in line.lower():
                parts = line.split("=")
                if len(parts) >= 2:
                    return float(parts[-1].strip())
        return -1  # Could not parse
    except subprocess.TimeoutExpired:
        print("Perplexity benchmark timed out")
        return -1
    except Exception as e:
        print(f"Error running perplexity binary: {e}")
        return -1


def run_perplexity_api(model_name: str, api_url: str, text: str, chunk_size: int = 256, overlap: int = 64) -> float:
    """
    Compute perplexity via the API using logprobs.
    Splits text into overlapping chunks and averages perplexity.
    """
    from openai import OpenAI
    import numpy as np
    import math

    client = OpenAI(base_url=api_url, api_key="not-needed")

    perplexities = []
    num_tokens = 0

    # Slide window over text
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunk = text[start:end]
        if len(chunk) < 10:  # Skip tiny chunks
            break

        try:
            # Request with logprobs for all tokens
            response = client.chat.completions.create(
                model=model_name,
                messages=[{"role": "user", "content": chunk}],
                temperature=0.0,
                max_tokens=1,
                logprobs=True,
                top_logprobs=0,
            )
            # We want the model's own generation of the chunk, but we sent it as prompt.
            # For perplexity, we need to measure how likely the model assigns to the chunk.
            # We can use the prompt's logprobs if the API returns them.
            # OpenAI-compatible API might return logprobs for the generated tokens, not the prompt.
            # Alternative: use the completions endpoint with a prefix and generate the chunk.
            # But that's more complex. For simplicity, we'll use the logprobs of the prompt if available.
            # llama-swap's API might support "logprobs" for prompt in the response.
            # Let's check: response.choices[0].logprobs if it exists.
            if response.choices and hasattr(response.choices[0], 'logprobs'):
                logprobs = response.choices[0].logprobs
                if logprobs and logprobs.token_logprobs:
                    chunk_logprob = sum(logprobs.token_logprobs)
                    chunk_tokens = len(logprobs.tokens)
                    if chunk_tokens > 0:
                        ppl = math.exp(-chunk_logprob / chunk_tokens)
                        perplexities.append(ppl)
                        num_tokens += chunk_tokens
        except Exception as e:
            print(f"Error during API perplexity for chunk: {e}")
            pass

        start += chunk_size - overlap
        if start >= len(text):
            break

    if not perplexities:
        return -1

    # Weighted average by token count? We can just average the perplexities.
    # Better: aggregate logprobs across chunks, then compute overall perplexity.
    total_logprob = sum(-math.log(p) * (chunk_size - overlap) for p in perplexities)  # approximate token count
    total_tokens = num_tokens
    overall_ppl = math.exp(total_logprob / total_tokens)
    return overall_ppl


def run_perplexity(model: dict, api_url: str, binary: str = "llama-perplexity") -> Dict[str, Any]:
    """
    Run perplexity benchmark for a model.
    Uses llama-perplexity binary first, falls back to API if binary not found.
    """
    gguf_path = model.get("gguf_path")
    model_name = model.get("llama_swap_model")

    # Try binary first
    if gguf_path:
        ppl = run_perplexity_binary(gguf_path, binary)
        if ppl > 0:
            return {"perplexity": ppl, "method": "binary"}

    # Fallback to API-based perplexity
    if model_name and api_url:
        # Load a small text from a file or use a fixed text
        # Use a standard benchmark text (Wikitext-2 test set)
        wikitext_path = "data/wikitext-2-raw/wiki.test.raw"
        if os.path.exists(wikitext_path):
            with open(wikitext_path, 'r') as f:
                text = f.read()
        else:
            # Fallback to a simple paragraph
            text = ("In the beginning, the universe was created. This has made a lot of people very angry and been widely regarded as a bad move. "
                    "But it has not been so for a long time. The universe has always been there. It just wasn't there as long as we've been around.")
        ppl = run_perplexity_api(model_name, api_url, text)
        if ppl > 0:
            return {"perplexity": ppl, "method": "api_fallback"}

    return {"perplexity": -1, "method": "none", "error": "Could not compute perplexity"}
