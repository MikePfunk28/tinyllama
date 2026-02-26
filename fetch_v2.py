"""
Fetch more data - improved categorization
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

# More diverse data sources
SOURCES = {
    "math": [
        ("gsm8k", "main", "question", "answer"),
    ],
    "code": [
        ("m-a-p/CodeFeedback-Filtered-Instruction", None, "query", "answer"),
    ],
    "reasoning": [
        ("openbookqa", "main", "question_stem", "choices"),
    ],
    "conversation": [
        ("Aeala/ShareGPT_Vicuna_unfiltered", None, "conversations", None),
    ],
    "knowledge": [
        ("garage-bAInd/Open-Platypus", None, "instruction", "output"),
        ("allenai/sciq", None, "question", "correct_answer"),
    ],
    "science": [
        ("allenai/sciq", None, "question", "correct_answer"),
    ],
}


def clean(t):
    if not t:
        return ""
    t = str(t)
    t = re.sub(r"<[^>]+>", "", t)
    t = re.sub(r"\s+", " ", t)
    return t.strip()


def format_sample(inst, out=None):
    inst = clean(inst)
    if not inst or len(inst) < 15:
        return None
    if out:
        out = clean(out)
        if len(out) < 3:
            out = None
    return f"Instruction: {inst}\nResponse: {out}" if out else f"Instruction: {inst}"


def categorize(text):
    t = text.lower()

    # Strong indicators
    if re.search(r"\d+\s*[\+\-\*\/\=]", t):
        return "math"
    if re.search(r"\bdef\s+\w+|class\s+\w+|import\s+\w+", t):
        return "code"
    if re.search(r"\bhello\b|\bhi\b|\bhow are\b|\bthank\b", t):
        return "conversation"
    if re.search(r"\bwhat is\b|\bwho is\b|\bdefine\b|\bexplain\b", t):
        return "knowledge"
    if re.search(r"\bscience\b|\bphysics\b|\bchemistry\b|\bbiology\b", t):
        return "science"
    if re.search(r"\bwhy\b|\bhow\b|\bcompare\b|\banalyze\b", t):
        return "reasoning"

    # Fallback based on source hints in the text
    if re.search(r"\bcalculate\b|\bsolve\b|\bequation\b", t):
        return "math"
    if re.search(r"\bfunction\b|\bprogram\b|\balgorithm\b", t):
        return "code"

    return None


def fetch(src, max_n=500):
    name, cfg, fin, fout = src
    samples = []

    try:
        print(f"    {name}...")
        ds = (
            load_dataset(name, cfg, split="train")
            if cfg
            else load_dataset(name, split="train")
        )

        for item in ds:
            if len(samples) >= max_n:
                break

            try:
                if fin == "conversations":
                    convs = item.get("conversations", [])
                    if isinstance(convs, list):
                        for msg in convs:
                            if isinstance(msg, dict):
                                role = msg.get("from", msg.get("role", ""))
                                content = msg.get("value", msg.get("content", ""))
                                if role in ["human", "user"] and content:
                                    # Conversation data - categorize as conversation
                                    s = format_sample(content)
                                    if s:
                                        samples.append(("conversation", s))
                                    break
                elif fin == "choices":
                    q = item.get("question_stem", "")
                    choices = item.get("choices", {})
                    if isinstance(choices, dict):
                        texts = choices.get("text", [])
                        if q and texts:
                            s = format_sample(f"{q} Options: {', '.join(texts)}")
                            if s:
                                samples.append(("reasoning", s))
                else:
                    inst = (
                        item.get(fin, "")
                        or item.get("instruction", "")
                        or item.get("question", "")
                    )
                    out = (
                        item.get(fout, "")
                        or item.get("output", "")
                        or item.get("answer", "")
                        or item.get("correct_answer", "")
                    )

                    if inst:
                        s = format_sample(inst, out if out else None)
                        if s:
                            cat = categorize(s)
                            # Use source hint if no clear category
                            if not cat:
                                if "sciq" in name:
                                    cat = "science"
                                elif "platypus" in name:
                                    cat = "knowledge"
                                elif "sharegpt" in name:
                                    cat = "conversation"
                                else:
                                    cat = None

                            if cat:
                                samples.append((cat, s))
            except:
                continue
    except Exception as e:
        print(f"      Err: {str(e)[:40]}")

    return samples


def main():
    print("=" * 50)
    print("FETCHING DATA")
    print("=" * 50)

    all_samples = defaultdict(list)

    for domain, srcs in SOURCES.items():
        print(f"\n[{domain.upper()}]")
        for src in srcs:
            samples = fetch(src, 400)
            for cat, sample in samples:
                if cat and len(sample) > 40 and len(sample) < 3000:
                    all_samples[cat].append(sample)
            print(f"    Got {len(samples)}")

    print("\n" + "=" * 50)
    print("SAMPLES")
    print("=" * 50)
    for d in EXPERTS:
        print(f"  {d}: {len(all_samples[d])}")

    print("\nMerging...")
    for d in EXPERTS:
        path = f"{DATA_DIR}/{d}_train.json"
        existing = []
        if os.path.exists(path):
            with open(path) as f:
                existing = json.load(f)

        combined = list(set(existing + all_samples[d]))
        random.shuffle(combined)

        n = len(combined)
        n_train = int(n * 0.8)
        n_val = int(n * 0.1)

        with open(f"{DATA_DIR}/{d}_train.json", "w") as f:
            json.dump(combined[:n_train], f)
        with open(f"{DATA_DIR}/{d}_val.json", "w") as f:
            json.dump(combined[n_train : n_train + n_val], f)
        with open(f"{DATA_DIR}/{d}_test.json", "w") as f:
            json.dump(combined[n_train + n_val :], f)

        print(f"  {d}: {len(existing)} + {len(all_samples[d])} -> {n}")

    print("\nDone!")


if __name__ == "__main__":
    main()
