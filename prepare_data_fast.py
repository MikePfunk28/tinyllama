import json
import os
import re
import random
from collections import defaultdict
import pandas as pd

random.seed(42)

DATA_DIR = "M:/tinyllama/cleaned_data"
os.makedirs(DATA_DIR, exist_ok=True)

EXPERTS = ["math", "code", "reasoning", "conversation", "knowledge"]
MAX_SAMPLES_PER_SOURCE = 5000


def clean_text(text):
    if not text:
        return ""
    text = str(text)
    text = re.sub(r"<[^>]+>", "", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def categorize_text(text):
    text_lower = text.lower()

    if any(
        p.search(text_lower)
        for p in [
            re.compile(r"\bsolve\b"),
            re.compile(r"\bequation\b"),
            re.compile(r"\bcalculate\b"),
            re.compile(r"\d+\s*[\+\-\*\/\=]\s*\d+"),
            re.compile(r"\bfind x\b"),
            re.compile(r"\bsum\b"),
            re.compile(r"\bintegral\b"),
            re.compile(r"\bderivative\b"),
            re.compile(r"\bprobability\b"),
        ]
    ):
        return "math"

    if any(
        p.search(text_lower)
        for p in [
            re.compile(r"\bdef\s+\w+"),
            re.compile(r"\bclass\s+\w+"),
            re.compile(r"\bimport\s+\w+"),
            re.compile(r"\bfunction\b"),
            re.compile(r"\breturn\b"),
            re.compile(r"\bcode\b"),
            re.compile(r"\balgorithm\b"),
            re.compile(r"\bprogram\b"),
            re.compile(r"\bapi\b"),
            re.compile(r"\bdebug\b"),
            re.compile(r"\.append\("),
            re.compile(r"\bfor\s+\w+\s+in\b"),
        ]
    ):
        return "code"

    if any(
        p.search(text_lower)
        for p in [
            re.compile(r"\bwhy\b"),
            re.compile(r"\bhow\b"),
            re.compile(r"\bexplain\b"),
            re.compile(r"\banalyze\b"),
            re.compile(r"\bcompare\b"),
            re.compile(r"\breason\b"),
            re.compile(r"\blogic\b"),
            re.compile(r"\bthink\b"),
            re.compile(r"\bconsider\b"),
            re.compile(r"\btherefore\b"),
            re.compile(r"\bbecause\b"),
            re.compile(r"\bwhat if\b"),
        ]
    ):
        return "reasoning"

    if any(
        p.search(text_lower)
        for p in [
            re.compile(r"\bhello\b"),
            re.compile(r"\bhi\b"),
            re.compile(r"\bhow are you\b"),
            re.compile(r"\bthank you\b"),
            re.compile(r"\bplease\b"),
            re.compile(r"\bhelp\s+me\b"),
            re.compile(r"\bcan you\b"),
            re.compile(r"\bchat\b"),
            re.compile(r"\badvice\b"),
            re.compile(r"\bi feel\b"),
            re.compile(r"\bsuggest\b"),
            re.compile(r"\bopinion\b"),
        ]
    ):
        return "conversation"

    if any(
        p.search(text_lower)
        for p in [
            re.compile(r"\bwhat is\b"),
            re.compile(r"\bwhat are\b"),
            re.compile(r"\bdefine\b"),
            re.compile(r"\bwho is\b"),
            re.compile(r"\bhistory\b"),
            re.compile(r"\bfact\b"),
            re.compile(r"\bdescribe\b"),
            re.compile(r"\btell me about\b"),
            re.compile(r"\blearn\b"),
            re.compile(r"\bscience\b"),
            re.compile(r"\btechnology\b"),
            re.compile(r"\bconcept\b"),
        ]
    ):
        return "knowledge"

    return None


def format_sample(inst, inp, out):
    inst = clean_text(inst)
    inp = clean_text(inp)
    out = clean_text(out)

    if inp and inp.strip():
        return f"Instruction: {inst}\nInput: {inp}\nResponse: {out}"
    return f"Instruction: {inst}\nResponse: {out}"


def process_file(path, extract_fn, source_name):
    print(f"  Processing {source_name}...")
    samples = defaultdict(list)
    count = 0

    try:
        df = pd.read_parquet(path)
        print(f"    Rows: {len(df)}")

        for _, row in df.iterrows():
            result = extract_fn(row)
            if result:
                inst, inp, out = result
                full_text = f"{inst} {inp} {out}"
                cat = categorize_text(full_text)
                if cat and len(samples[cat]) < MAX_SAMPLES_PER_SOURCE:
                    formatted = format_sample(inst, inp, out)
                    if len(formatted) > 50 and len(formatted) < 4000:
                        samples[cat].append(formatted)
                        count += 1
                        if count % 1000 == 0:
                            print(f"    Categorized: {count}")
    except Exception as e:
        print(f"    Error: {e}")

    return samples


def extract_openthoughts(row):
    try:
        convs = row.get("conversations", [])
        if isinstance(convs, list) and len(convs) >= 2:
            user_msg = ""
            asst_msg = ""
            for msg in convs:
                if msg.get("role") == "user":
                    user_msg = msg.get("content", "")
                elif msg.get("role") == "assistant":
                    asst_msg = msg.get("content", "")
            if user_msg and asst_msg:
                return (user_msg, "", asst_msg)
    except:
        pass
    return None


def extract_opencode(row):
    inp = row.get("input", "")
    out = row.get("output", "")
    sol = row.get("solution", "")
    if inp and (out or sol):
        return (inp, "", out or sol)
    return None


def main():
    print("=" * 60)
    print("FAST DATA PREPARATION")
    print("=" * 60)

    all_samples = defaultdict(list)

    # OpenThoughts - just first file
    path = "M:/training_models/OpenThoughts-114k/data/train-00000-of-00006.parquet"
    if os.path.exists(path):
        samples = process_file(path, extract_openthoughts, "OpenThoughts")
        for cat, data in samples.items():
            all_samples[cat].extend(data)

    # OpenCodeReasoning - just first file
    path = "M:/OpenCodeReasoning/split_0/train-00000-of-00030.parquet"
    if os.path.exists(path):
        samples = process_file(path, extract_opencode, "OpenCodeReasoning")
        for cat, data in samples.items():
            all_samples[cat].extend(data)

    # Corporate dataset
    print("  Processing Corporate Dataset...")
    corp_path = "M:/datasets--MikePfunk28--corporateDataset/snapshots/eb62b89763e6859769d34da13bc1b3b7d4b9d15c/clean_corporate_dataset.jsonl"
    if os.path.exists(corp_path):
        count = 0
        with open(corp_path, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    data = json.loads(line)
                    inst = data.get("instruction", "")
                    inp = data.get("input", "")
                    out = data.get("output", "")
                    if inst and out:
                        full_text = f"{inst} {inp} {out}"
                        cat = categorize_text(full_text)
                        if cat and len(all_samples[cat]) < MAX_SAMPLES_PER_SOURCE * 2:
                            formatted = format_sample(inst, inp, out)
                            if len(formatted) > 50 and len(formatted) < 4000:
                                all_samples[cat].append(formatted)
                                count += 1
                except:
                    pass
        print(f"    Corporate samples: {count}")

    print("\nSamples per category:")
    for cat in EXPERTS:
        print(f"  {cat}: {len(all_samples[cat])}")

    # Save with splits
    print("\nSaving with train/val/test splits...")
    for cat in EXPERTS:
        samples = all_samples[cat]
        random.shuffle(samples)

        n = len(samples)
        n_train = int(n * 0.8)
        n_val = int(n * 0.1)

        splits = {
            "train": samples[:n_train],
            "val": samples[n_train : n_train + n_val],
            "test": samples[n_train + n_val :],
        }

        for split_name, data in splits.items():
            path = f"{DATA_DIR}/{cat}_{split_name}.json"
            with open(path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)
            print(f"  {cat}_{split_name}: {len(data)}")

    # Router training data
    print("\nCreating router data...")
    router_data = []
    for cat in EXPERTS:
        for sample in all_samples[cat][:3000]:
            router_data.append({"text": sample[:500], "label": cat})

    random.shuffle(router_data)
    n = len(router_data)
    n_train = int(n * 0.8)

    with open(f"{DATA_DIR}/router_train.json", "w") as f:
        json.dump(router_data[:n_train], f, indent=2)
    with open(f"{DATA_DIR}/router_val.json", "w") as f:
        json.dump(router_data[n_train:], f, indent=2)

    print(f"  Router train: {n_train}, val: {n - n_train}")
    print("\nDone!")


if __name__ == "__main__":
    main()
