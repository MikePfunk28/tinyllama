"""
Fast Quality Check - Quick evaluation
"""

import os
import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

BASE = "M:/training_models/gemma-3-270m-it"
EXP = "M:/tinyllama/experts"
DOMAINS = ["math", "code", "reasoning", "conversation", "knowledge", "science"]

TESTS = {
    "math": [("What is 7 + 8?", ["15"]), ("5 * 4 = ?", ["20"])],
    "code": [("def hello():", ["def", "print"]), ("Write Python code", ["python"])],
    "reasoning": [
        ("Why is sky blue?", ["scatter", "light"]),
        ("How does X work?", ["work"]),
    ],
    "conversation": [("Hello!", ["hello", "hi"]), ("How are you?", ["good", "help"])],
    "knowledge": [("What is DNA?", ["genetic"]), ("Capital of France?", ["paris"])],
    "science": [("What is H2O?", ["water"]), ("Speed of light?", ["300", "299"])],
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
                inp["input_ids"], max_new_tokens=20, pad_token_id=tok.pad_token_id
            )
        resp = tok.decode(out[0], skip_special_tokens=True).lower()
        found = sum(1 for k in keywords if k.lower() in resp)
        scores.append(found / len(keywords))

    q = sum(scores) / len(scores) if scores else 0
    results[domain] = q
    print(f"  {domain}: {q * 100:.0f}%")

avg = sum(results.values()) / len(results)
print(f"\nAvg: {avg * 100:.0f}%")

with open(f"{EXP}/quality_results.json", "w") as f:
    json.dump({"results": results, "average": avg}, f)

print(f"Saved: {EXP}/quality_results.json")
