"""
Download and prepare new datasets from HuggingFace for MoE training
"""

import os
import json
import re
from datasets import load_dataset
from collections import defaultdict
import random

random.seed(42)

DATA_DIR = "M:/tinyllama/cleaned_data"
EXPERTS = ["math", "code", "reasoning", "conversation", "knowledge", "science"]

DATASETS = {
    "math": [
        ("gsm8k", "main", "question", "answer"),
    ],
    "code": [
        ("m-a-p/CodeFeedback-Filtered-Instruction", None, "query", "answer"),
    ],
    "conversation": [
        ("Aeala/ShareGPT_Vicuna_unfiltered", None, "conversations", None),
    ],
    "knowledge": [
        ("garage-bAInd/Open-Platypus", None, "instruction", "output"),
    ],
    "science": [
        ("sciq", None, "question", "correct_answer"),
    ],
}


def clean_text(text):
    if not text:
        return ""
    text = str(text)
    text = re.sub(r"<[^>]+>", "", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def categorize_text(text, source_hint=None):
    text_lower = text.lower()

    if source_hint:
        if "sharegpt" in source_hint.lower() or "vicuna" in source_hint.lower():
            return "conversation"
        if "platypus" in source_hint.lower():
            return "knowledge"

    knowledge_patterns = [
        r"^what\b",
        r"^who\b",
        r"^where\b",
        r"^when\b",
        r"^why\b",
        r"^how\b",
        r"\bexplain\b",
        r"\bdefine\b",
        r"\bhistory\b",
        r"\bscience\b",
        r"\blearn\b",
        r"\bknow\b",
        r"\bunderstand\b",
        r"\btell\b",
        r"\bdescribe\b",
        r"\bfact\b",
        r"\binformation\b",
        r"\bwhat is\b",
        r"\bwhat are\b",
        r"\bwho is\b",
        r"\bwhere is\b",
        r"\bwhen did\b",
    ]

    if any(p.search(text_lower) for p in [re.compile(p) for p in knowledge_patterns]):
        return "knowledge"

    reasoning_patterns = [
        r"\bwhy\b",
        r"\bhow\b",
        r"\bcompare\b",
        r"\banalyze\b",
        r"\bif\b.*\bthen\b",
        r"\bcause\b",
        r"\beffect\b",
        r"\breason\b",
        r"\bdeduce\b",
        r"\binfer\b",
    ]

    if any(p.search(text_lower) for p in [re.compile(p) for p in reasoning_patterns]):
        return "reasoning"

    conversation_patterns = [
        r"\bhello\b",
        r"\bhi\b",
        r"\bhey\b",
        r"\bhow\b",
        r"\bthank\b",
        r"\bplease\b",
        r"\bhelp\b",
        r"\byou\b",
        r"\bme\b",
        r"\bmy\b",
        r"\bwould\b",
        r"\bcould\b",
        r"\bcan\b",
        r"\bi\b",
        r"\bwe\b",
        r"\btalk\b",
        r"\bchat\b",
        r"\bassistant\b",
    ]

    if any(
        p.search(text_lower) for p in [re.compile(p) for p in conversation_patterns]
    ):
        return "conversation"

    math_patterns = [
        r"\bcalculate\b",
        r"\bcompute\b",
        r"\bsolve\b",
        r"\bequation\b",
        r"\bmath\b",
        r"\bnumber\b",
        r"\bplus\b",
        r"\bminus\b",
        r"\btimes\b",
        r"\bdivided\b",
        r"\b\d+\b",
        r"\bgeometry\b",
        r"\balgebra\b",
        r"\bprobability\b",
    ]

    if any(p.search(text_lower) for p in [re.compile(p) for p in math_patterns]):
        return "math"

    code_patterns = [
        r"\bcode\b",
        r"\bfunction\b",
        r"\bprogramming\b",
        r"\bpython\b",
        r"\bjavascript\b",
        r"\bjava\b",
        r"\bclass\b",
        r"\bdef\b",
        r"\bimport\b",
        r"\bapi\b",
        r"\bdebug\b",
        r"\balgorithm\b",
        r"\bsoftware\b",
    ]

    if any(p.search(text_lower) for p in [re.compile(p) for p in code_patterns]):
        return "code"

    science_patterns = [
        r"\bscience\b",
        r"\bphysics\b",
        r"\bchemistry\b",
        r"\bbiology\b",
        r"\bexperiment\b",
        r"\bhypothesis\b",
        r"\btheory\b",
        r"\bdata\b",
        r"\bresearch\b",
        r"\bcell\b",
        r"\bgene\b",
        r"\batom\b",
        r"\bmolecule\b",
    ]

    if any(p.search(text_lower) for p in [re.compile(p) for p in science_patterns]):
        return "science"

    return None


def format_sample(instruction, output=None):
    instruction = clean_text(instruction)
    if output:
        output = clean_text(output)
        return f"Instruction: {instruction}\nResponse: {output}"
    return f"Instruction: {instruction}"


def download_dataset(name, config, field_in, field_out, max_samples=500):
    print(f"  Downloading {name}...")
    try:
        if config:
            ds = load_dataset(name, config, split="train", trust_remote_code=True)
        else:
            ds = load_dataset(name, split="train", trust_remote_code=True)

        samples = []
        for item in ds:
            if len(samples) >= max_samples:
                break

            if field_in == "conversations" and isinstance(
                item.get("conversations"), list
            ):
                for msg in item["conversations"]:
                    if isinstance(msg, dict) and msg.get("from") == "human":
                        instruction = msg.get("value", "")
                        if instruction and len(instruction) > 20:
                            samples.append(format_sample(instruction))
            else:
                instruction = item.get(field_in, "") or item.get("instruction", "")
                output = (
                    item.get(field_out, "")
                    or item.get("output", "")
                    or item.get("answer", "")
                )

                if instruction and len(instruction) > 20:
                    samples.append(
                        format_sample(instruction, output if output else None)
                    )

        print(f"    Got {len(samples)} samples")
        return samples
    except Exception as e:
        print(f"    Error: {e}")
        return []


def main():
    print("=" * 60)
    print("DOWNLOADING NEW DATASETS")
    print("=" * 60)

    all_samples = defaultdict(list)

    for expert, datasets in DATASETS.items():
        print(f"\n[{expert.upper()}]")
        for dataset_info in datasets:
            name, config, field_in, field_out = dataset_info
            samples = download_dataset(
                name, config, field_in, field_out, max_samples=500
            )

            for sample in samples:
                # Use source hint for categorization
                if "sharegpt" in name.lower() or "vicuna" in name.lower():
                    category = "conversation"
                elif "platypus" in name.lower():
                    category = "knowledge"
                else:
                    category = categorize_text(sample)

                if category and len(all_samples[category]) < 2000:
                    if len(sample) > 50 and len(sample) < 4000:
                        all_samples[category].append(sample)

    print("\n" + "=" * 60)
    print("SAMPLES PER CATEGORY")
    print("=" * 60)
    for cat in EXPERTS:
        print(f"  {cat}: {len(all_samples[cat])}")

    print("\nMerging with existing data...")
    for cat in EXPERTS:
        existing_path = f"{DATA_DIR}/{cat}_train.json"
        if os.path.exists(existing_path):
            with open(existing_path) as f:
                existing = json.load(f)
            combined = list(set(existing + all_samples[cat]))
            random.shuffle(combined)

            n = len(combined)
            n_train = int(n * 0.8)
            n_val = int(n * 0.1)

            with open(f"{DATA_DIR}/{cat}_train.json", "w") as f:
                json.dump(combined[:n_train], f)
            with open(f"{DATA_DIR}/{cat}_val.json", "w") as f:
                json.dump(combined[n_train : n_train + n_val], f)
            with open(f"{DATA_DIR}/{cat}_test.json", "w") as f:
                json.dump(combined[n_train + n_val :], f)

            print(
                f"  {cat}: {len(existing)} + {len(all_samples[cat])} -> {len(combined)} total"
            )

    print("\nDone!")


if __name__ == "__main__":
    main()
