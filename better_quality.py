"""
Better Quality Check - More test questions
"""

import os
import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

BASE = "M:/training_models/gemma-3-270m-it"
EXP = "M:/tinyllama/experts"
DOMAINS = ["math", "code", "reasoning", "conversation", "knowledge", "science"]

# More test questions per domain (5 each)
TESTS = {
    "math": [
        ("What is 7 + 8?", ["15"]),
        ("5 * 4 = ?", ["20"]),
        ("10 - 3 = ?", ["7"]),
        ("What is 2 + 2?", ["4"]),
        ("Calculate 6 * 7", ["42"]),
    ],
    "code": [
        ("def hello():", ["def"]),
        ("Write Python code", ["python", "code"]),
        ("How to print in Python?", ["print"]),
        ("Create a variable", ["="]),
        ("for loop example", ["for"]),
    ],
    "reasoning": [
        ("Why is sky blue?", ["scatter", "light"]),
        ("How does rain form?", ["water", "cloud"]),
        ("Why do birds fly?", ["wing", "fly"]),
        ("How does engine work?", ["fuel", "energy"]),
        ("Why do we sleep?", ["rest", "body"]),
    ],
    "conversation": [
        ("Hello!", ["hello", "hi", "hey"]),
        ("How are you?", ["good", "well", "fine"]),
        ("Thank you", ["welcome", "happy"]),
        ("Good morning", ["morning", "good"]),
        ("Nice to meet you", ["nice", "meet"]),
    ],
    "knowledge": [
        ("What is DNA?", ["genetic", "gene"]),
        ("Capital of France?", ["paris"]),
        ("What is photosynthesis?", ["plant", "sun"]),
        ("Who wrote Shakespeare?", ["shakespeare"]),
        ("What is gravity?", ["force", "mass"]),
    ],
    "science": [
        ("What is H2O?", ["water"]),
        ("Speed of light?", ["300", "299"]),
        ("What is atom?", ["particle", "electron"]),
        ("Solar system?", ["planet", "sun"]),
        ("What is energy?", ["work", "power"]),
    ],
}

print("Loading model...")
tok = AutoTokenizer.from_pretrained(BASE)
tok.pad_token = tok.eos_token
model = AutoModelForCausalLM.from_pretrained(BASE, torch_dtype=torch.float32)

results = {}
for domain in DOMAINS:
    tests = TESTS.get(domain, [])
    scores = []

    for q, keywords in tests:
        inp = tok(f"{q}\nAnswer:", return_tensors="pt")
        with torch.no_grad():
            out = model.generate(
                inp["input_ids"], max_new_tokens=25, pad_token_id=tok.pad_token_id
            )
        resp = tok.decode(out[0], skip_special_tokens=True).lower()
        found = sum(1 for k in keywords if k.lower() in resp)
        scores.append(found / len(keywords) if keywords else 0)

    q = sum(scores) / len(scores) if scores else 0
    results[domain] = q
    status = "OK" if q >= 0.95 else "LOW"
    print(f"  {domain}: {q * 100:.0f}% [{status}]")

avg = sum(results.values()) / len(results)
print(f"\nAvg: {avg * 100:.0f}%")

with open(f"{EXP}/quality_results.json", "w") as f:
    json.dump({"results": results, "average": avg}, f)

print(f"Saved")
