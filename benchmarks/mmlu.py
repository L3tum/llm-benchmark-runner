#!/usr/bin/env python3
"""
MMLU benchmark for models via llama-swap API.
Evaluates a subset of the MMLU benchmark.
"""

import os
import json
from typing import List, Dict, Any, Optional
from openai import OpenAI
import requests
from tqdm import tqdm

# MMLU subject list (for full benchmark) - we'll use a small subset for quick evaluation
MMLU_SUBJECTS = [
    "abstract_algebra", "anatomy", "astronomy", "business_ethics", "clinical_knowledge",
    "college_biology", "college_chemistry", "college_computer_science", "college_mathematics",
    "college_medicine", "college_physics", "computer_security", "conceptual_physics",
    "econometrics", "electrical_engineering", "elementary_mathematics", "formal_logic",
    "global_facts", "high_school_biology", "high_school_chemistry", "high_school_computer_science",
    "high_school_european_history", "high_school_geography", "high_school_government_and_politics",
    "high_school_macroeconomics", "high_school_mathematics", "high_school_microeconomics",
    "high_school_physics", "high_school_psychology", "high_school_statistics", "high_school_us_history",
    "high_school_world_history", "human_aging", "human_sexuality", "international_law",
    "jurisprudence", "logical_fallacies", "machine_learning", "management", "marketing",
    "medical_genetics", "miscellaneous", "moral_disputes", "moral_scenarios", "nutrition",
    "philosophy", "prehistory", "professional_accounting", "professional_law",
    "professional_medicine", "professional_psychology", "public_relations", "security_studies",
    "sociology", "us_foreign_policy", "virology", "world_religions"
]

# For quick mode, we pick a few subjects and 20-30 samples each
QUICK_SUBJECTS = [
    "high_school_mathematics", "high_school_physics", "high_school_computer_science",
    "college_mathematics", "prehistory", "miscellaneous"
]


def fetch_mmlu_subset(api_url: str = None, num_samples: int = 100, subset_path: Optional[str] = None) -> List[dict]:
    """
    Fetch a subset of MMLU questions.
    If subset_path is provided, load from file. Otherwise, fetch from a public source.
    """
    if subset_path and os.path.exists(subset_path):
        with open(subset_path, 'r') as f:
            data = json.load(f)
            if isinstance(data, list):
                return data[:num_samples]
            elif isinstance(data, dict) and 'data' in data:
                return data['data'][:num_samples]
            else:
                raise ValueError("Unexpected JSON structure in subset file.")

    # Try to fetch from a public MMLU dataset (e.g., from github raw)
    # We'll use a small curated subset from Hugging Face datasets or similar.
    # For robustness, we'll use a well-known quick MMLU subset URL.
    url = "https://raw.githubusercontent.com/hendrycks/test/master/data/mmlu/"
    subjects = QUICK_SUBJECTS
    questions = []
    for subj in subjects:
        train_url = f"{url}/{subj}/dev.csv"
        test_url = f"{url}/{subj}/test.csv"
        for csv_url in [test_url, train_url]:
            try:
                response = requests.get(csv_url, timeout=10)
                if response.status_code == 200:
                    lines = response.text.strip().split('\n')
                    if len(lines) < 2:
                        continue
                    header = lines[0].split(',')
                    for line in lines[1:]:
                        # Simple CSV parsing (no quotes handling for now)
                        fields = [f.strip() for f in line.split(',')]
                        if len(fields) >= 5:
                            q = fields[0]
                            options = []
                            for opt_idx in range(1, 5):
                                options.append(fields[opt_idx])
                            answer = fields[5]  # letter answer
                            questions.append({
                                "subject": subj,
                                "question": q,
                                "options": options,
                                "answer": answer.strip()
                            })
                            if len(questions) >= num_samples:
                                return questions
            except Exception:
                pass
    return questions[:num_samples]


def run_mmlu(model: dict, api_url: str, mmlu_config: dict) -> Dict[str, Any]:
    """
    Run MMLU benchmark on a model.
    """
    num_samples = mmlu_config.get("num_samples", 100)
    subset_path = mmlu_config.get("subset_path")

    # Get model name and API
    model_name = model.get("llama_swap_model")
    if not model_name:
        return {"error": "Model name not specified"}

    client = OpenAI(base_url=api_url, api_key="not-needed")

    # Fetch questions
    questions = fetch_mmlu_subset(api_url, num_samples, subset_path)
    if not questions:
        return {"error": "Could not fetch MMLU questions"}

    # Prepare prompt
    system_prompt = "You are a helpful assistant that answers multiple-choice questions. Only output the letter of the correct option."

    import time
    total_correct = 0
    results_by_subject = {}
    question_times = []  # time per question in seconds

    for q in tqdm(questions, desc=f"MMLU on {model_name}"):
        # Format options
        options_text = "\n".join([f"{chr(65+i)}: {opt}" for i, opt in enumerate(q["options"])])
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"{q['question']}\n\n{options_text}"}
        ]

        start = time.time()
        try:
            response = client.chat.completions.create(
                model=model_name,
                messages=messages,
                temperature=0.0,
                max_tokens=5,
            )
            answer = response.choices[0].message.content.strip().upper()
            # Extract first letter if present
            predicted = answer[0] if answer else ""
            if predicted in "ABCD":
                is_correct = (predicted == q["answer"].strip())
                if is_correct:
                    total_correct += 1
                subj = q.get("subject", "unknown")
                results_by_subject[subj] = results_by_subject.get(subj, {"correct": 0, "total": 0})
                results_by_subject[subj]["total"] += 1
                if is_correct:
                    results_by_subject[subj]["correct"] += 1
            else:
                # If the model outputs something that's not a letter, count as wrong
                subj = q.get("subject", "unknown")
                results_by_subject[subj] = results_by_subject.get(subj, {"correct": 0, "total": 0})
                results_by_subject[subj]["total"] += 1
        except Exception as e:
            print(f"Error on question: {e}")
            continue
        elapsed = time.time() - start
        question_times.append(elapsed)

    accuracy = total_correct / len(questions) if questions else 0
    avg_time_per_question = np.mean(question_times) if question_times else 0
    total_time = np.sum(question_times)
    questions_per_second = len(questions) / total_time if total_time > 0 else 0

    return {
        "mmlu_accuracy": accuracy,
        "total_questions": len(questions),
        "correct": total_correct,
        "results_by_subject": results_by_subject,
        "mmlu_speed": {
            "avg_time_per_question_s": avg_time_per_question,
            "total_time_s": total_time,
            "questions_per_second": questions_per_second,
        }
    }
