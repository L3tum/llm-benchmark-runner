#!/usr/bin/env python3
"""
MMLU-Pro benchmark — modeled after the original MMLU-Pro eval script.
Loads from HuggingFace, handles 10 options, CoT few-shot, per-subject accuracy.
"""

import json
import os
import random
import re
import time
from typing import Dict, Any, List, Optional
from openai import OpenAI
from datasets import load_dataset
from tqdm import tqdm


def run_mmlu_pro(model: dict, config: dict) -> Dict[str, Any]:
    """
    Run MMLU-Pro benchmark on model.
    Args:
        model: model dict with display_name, proxy, cmdStop
        config: mmlu_pro config dict (num_samples, subjects)
    Returns:
        dict with accuracy, results_by_subject, etc.
    """

    # Load datasets
    dataset = load_dataset("TIGER-Lab/MMLU-Pro")
    test_df = dataset["test"]
    val_df = dataset["validation"]

    # Preprocess: group by category, filter N/A options
    def preprocess(df):
        res = {}
        for item in df:
            opts = [o for o in item["options"] if o != "N/A"]
            item["options"] = opts
            cat = item["category"]
            if cat not in res:
                res[cat] = []
            res[cat].append(item)
        return res

    test_data = preprocess(test_df)
    val_data = preprocess(val_df)

    # Config
    num_samples = config.get("num_samples")
    subjects = config.get("subjects")
    if isinstance(subjects, str):
        subjects = [s.strip() for s in subjects.split(",")]
    elif subjects is None:
        subjects = list(test_data.keys())

    # OpenAI-compatible client
    proxy = model["proxy"]
    client = OpenAI(base_url=proxy, api_key="not-needed")

    # Category record for per-subject accuracy
    category_record: Dict[str, Dict[str, float]] = {}
    total_results: List[dict] = []

    for category in subjects:
        if category not in test_data:
            print(f"Category {category} not found in MMLU-Pro")
            continue

        test_questions = test_data[category]
        if num_samples:
            test_questions = test_questions[:num_samples]

        # Get few-shot examples from validation set
        cot_examples = val_data.get(category, [])[:5]  # up to 5 examples

        print(f"\nEvaluating {category}: {len(test_questions)} questions")

        category_correct = 0
        category_total = 0

        for q in tqdm(test_questions, desc=category):
            # Build prompt with CoT examples
            prompt = f"The following are multiple choice questions (with answers) about {category}. " \
                     f"Think step by step and then output the answer in the format of \"The answer is (X)\" at the end.\n\n"

            for ex in cot_examples:
                prompt += format_example(ex["question"], ex["options"], ex.get("cot_content", ""))

            # Current question
            prompt += format_example(q["question"], q["options"])

            try:
                start = time.time()
                response = client.chat.completions.create(
                    model="model",  # llama-server style
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.0,
                    max_tokens=4000,
                )
                answer_text = response.choices[0].message.content.strip()
                elapsed = time.time() - start
            except Exception as e:
                print(f"\n  Error on question: {e}")
                category_total += 1
                # Random guess if error
                category_record[category] = category_record.get(category, {"corr": 0.0, "wrong": 0.0})
                category_record[category]["wrong"] += 1
                continue

            # Extract answer
            pred = extract_answer(answer_text)
            if pred is None:
                pred = random_guess(len(q["options"]))

            is_correct = pred == q["answer"]
            if is_correct:
                category_correct += 1
            category_total += 1

            # Save result
            result_item = {
                "question_id": q["question_id"],
                "question": q["question"],
                "category": category,
                "pred": pred,
                "answer": q["answer"],
                "answer_index": q["answer_index"],
                "model_output": answer_text,
                "correct": is_correct,
                "time_s": elapsed,
            }
            total_results.append(result_item)

        # Update category record
        category_record[category] = {"corr": category_correct, "wrong": category_total - category_correct}

    # Calculate overall accuracy
    total_correct = sum(r.get("corr", 0) for r in category_record.values())
    total_wrong = sum(r.get("wrong", 0) for r in category_record.values())
    overall_accuracy = total_correct / (total_correct + total_wrong) if (total_correct + total_wrong) > 0 else 0

    # Calculate per-category accuracy
    for cat in category_record:
        c = category_record[cat]["corr"]
        w = category_record[cat]["wrong"]
        category_record[cat]["acc"] = c / (c + w) if (c + w) > 0 else 0

    return {
        "accuracy": overall_accuracy,
        "results_by_subject": category_record,
        "total_questions": len(total_results),
        "results": total_results,
    }


def random_guess(num_options: int) -> str:
    """Random guess from A-J."""
    choice_map = "ABCDEFGHIJ"
    return choice_map[random.randint(0, num_options - 1)]


def format_example(question: str, options: list, cot_content: str = "") -> str:
    """Format a question with options for the prompt."""
    if cot_content == "":
        cot_content = "Let's think step by step."
    if cot_content.startswith("A: "):
        cot_content = cot_content[3:]
    example = f"Question: {question}\nOptions: "
    choice_map = "ABCDEFGHIJ"
    for i, opt in enumerate(options):
        example += f"{choice_map[i]}. {opt}\n"
    if cot_content == "":
        example += "Answer: "
    else:
        example += f"Answer: {cot_content}\n\n"
    return example


def extract_answer(text: str) -> Optional[str]:
    """Extract answer from model output (A-J). Only checks the last line."""
    # Only look at the last line to avoid false positives from reasoning
    last_line = text.strip().split('\n')[-1] if text else ""

    # Primary pattern: answer is (X)
    match = re.search(r"answer is \(?([A-J])\)?", last_line, re.IGNORECASE)
    if match:
        return match.group(1).upper()

    # Fallback: answer: X
    match = re.search(r"[aA]nswer:\s*([A-J])", last_line)
    if match:
        return match.group(1).upper()

    # Final: last single letter from A-J (most likely guess)
    match = re.search(r"\b([A-J])\b(?!\s*[,;:]\s*\b[A-J]\b)", last_line)
    if match:
        return match.group(1).upper()

    return None