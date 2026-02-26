"""
Quality Evaluator - Measures adapter performance
Goal: 95% quality on each domain
"""

import os
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
import re

BASE_MODEL = "M:/training_models/gemma-3-270m-it"
EXPERTS_DIR = "M:/tinyllama/experts"
DATA_DIR = "M:/tinyllama/cleaned_data"
DOMAINS = ["math", "code", "reasoning", "conversation", "knowledge", "science"]

# Test questions per domain with expected keywords
TEST_QUESTIONS = {
    "math": [
        ("What is 7 + 8?", ["15"]),
        ("Calculate 20 * 3", ["60"]),
        ("What is 100 - 45?", ["55"]),
        ("Solve: 5 + 5 = ?", ["10"]),
        ("What is 12 / 4?", ["3"]),
    ],
    "code": [
        ("Write a Python hello world", ["print", "hello", "world"]),
        ("Define a function in Python", ["def", "function"]),
        ("How do you create a list in Python?", ["list", "[]", "["]),
        ("Write a for loop", ["for", "in", "range"]),
        ("Create a variable in Python", ["=", "variable"]),
    ],
    "reasoning": [
        ("Why is the sky blue?", ["rayleigh", "scattering", "light"]),
        ("How does photosynthesis work?", ["plant", "sunlight", "energy"]),
        ("Why do birds migrate?", ["food", "weather", "winter"]),
        ("Explain cause and effect", ["cause", "effect", "because"]),
        ("How do computers work?", ["process", "data", "binary"]),
    ],
    "conversation": [
        ("Hello, how are you?", ["hello", "hi", "help", "good"]),
        ("Thank you for your help", ["welcome", "happy", "help"]),
        ("Good morning!", ["morning", "good", "hello"]),
        ("Can you help me?", ["help", "yes", "sure"]),
        ("Nice to meet you", ["meet", "nice", "pleasure"]),
    ],
    "knowledge": [
        ("What is photosynthesis?", ["plant", "sunlight", "energy"]),
        ("Who invented the telephone?", ["bell", "telephone"]),
        ("What is the capital of France?", ["paris"]),
        ("Define gravity", ["force", "mass", "attraction"]),
        ("What is machine learning?", ["data", "learn", "algorithm"]),
    ],
    "science": [
        ("What is H2O?", ["water", "hydrogen", "oxygen"]),
        ("Explain the water cycle", ["evaporation", "condensation", "rain"]),
        ("What is DNA?", ["genetic", "double", "helix"]),
        ("How does electricity work?", ["electron", "flow", "current"]),
        ("What is the speed of light?", ["299", "300", "million"]),
    ],
}


class LoRALayer(nn.Module):
    def __init__(self, orig, r=8, a=16):
        super().__init__()
        self.orig, self.r, self.a = orig, r, a
        self.A = nn.Parameter(torch.zeros(r, orig.in_features))
        self.B = nn.Parameter(torch.zeros(orig.out_features, r))
        nn.init.kaiming_uniform_(self.A, a=5**0.5)
        for p in orig.parameters():
            p.requires_grad = False

    def forward(self, x):
        return self.orig(x) + (self.a / self.r) * (x @ self.A.T @ self.B.T)


def load_adapter(model, domain):
    """Load LoRA adapter into model"""
    path = f"{EXPERTS_DIR}/{domain}_lora_adapter.pt"
    if not os.path.exists(path):
        return False

    data = torch.load(path, map_location="cpu", weights_only=False)
    state = data.get("lora_state_dict", {})

    loaded = 0
    for name, module in model.named_modules():
        if any(x in name for x in ["q_proj", "v_proj"]) and isinstance(
            module, nn.Linear
        ):
            a_key = f"{name}.lora_A"
            b_key = f"{name}.lora_B"
            if a_key in state and b_key in state:
                parts = name.rsplit(".", 1)
                if len(parts) == 2:
                    parent = model
                    for p in parts[0].split("."):
                        if p:
                            parent = getattr(parent, p)
                    lora = LoRALayer(module)
                    lora.A.data = state[a_key]
                    lora.B.data = state[b_key]
                    setattr(parent, parts[1], lora)
                    loaded += 1

    return loaded > 0


def evaluate_domain(model, tok, domain, device="cpu"):
    """Evaluate a domain's quality"""
    questions = TEST_QUESTIONS.get(domain, [])
    if not questions:
        return 0.0

    scores = []

    for question, expected_keywords in questions:
        prompt = f"{question}\n\nAnswer:"
        inputs = tok(prompt, return_tensors="pt")
        ids = inputs["input_ids"].to(device)

        with torch.no_grad():
            out = model.generate(
                ids,
                max_new_tokens=50,
                temperature=0.7,
                do_sample=True,
                top_k=40,
                pad_token_id=tok.pad_token_id,
            )

        response = tok.decode(out[0], skip_special_tokens=True).lower()

        # Check for expected keywords
        found = sum(1 for kw in expected_keywords if kw.lower() in response)
        score = found / len(expected_keywords)
        scores.append(score)

    return sum(scores) / len(scores) if scores else 0.0


def main():
    print("=" * 60)
    print("QUALITY EVALUATION")
    print("=" * 60)

    device = torch.device("cpu")

    tok = AutoTokenizer.from_pretrained(BASE_MODEL)
    tok.pad_token = tok.eos_token

    results = {}

    for domain in DOMAINS:
        print(f"\n[{domain.upper()}]")

        # Load fresh model
        model = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL, torch_dtype=torch.float32
        )
        model = model.to(device)

        # Load adapter
        if load_adapter(model, domain):
            print(f"  Adapter loaded")

            # Evaluate
            quality = evaluate_domain(model, tok, domain, device)
            results[domain] = quality

            status = "OK" if quality >= 0.95 else "NEEDS TRAINING"
            print(f"  Quality: {quality * 100:.1f}% [{status}]")
        else:
            print(f"  No adapter found")
            results[domain] = 0.0

        del model

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    avg_quality = sum(results.values()) / len(results) if results else 0

    for domain, quality in sorted(results.items(), key=lambda x: -x[1]):
        bar = "█" * int(quality * 20) + "░" * (20 - int(quality * 20))
        print(f"  {domain:12} [{bar}] {quality * 100:5.1f}%")

    print(f"\n  Average: {avg_quality * 100:.1f}%")
    print(f"  Goal: 95%")

    # Save results
    results_path = f"{EXPERTS_DIR}/quality_results.json"
    with open(results_path, "w") as f:
        json.dump(
            {
                "results": results,
                "average": avg_quality,
                "goal": 0.95,
                "domains_needing_training": [d for d, q in results.items() if q < 0.95],
            },
            f,
            indent=2,
        )

    print(f"\n  Saved: {results_path}")

    return results


if __name__ == "__main__":
    main()
