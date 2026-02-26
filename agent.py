"""
MoE Agent - Fully Functional
Routes to expert, applies LoRA, generates real responses
"""

import os
import re
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

try:
    import torch_directml

    device = torch.device("cpu")  # Force CPU for stability
except:
    device = torch.device("cpu")

BASE_MODEL = "M:/training_models/gemma-3-270m-it"
ADAPTERS_DIR = "M:/tinyllama/experts"
EXPERTS = ["math", "code", "reasoning", "conversation", "knowledge"]


class LoRAManager:
    def __init__(self, model):
        self.model = model
        self.adapters = {}
        self.base_weights = {}
        self._save_base()

    def _save_base(self):
        for name, param in self.model.named_parameters():
            if "q_proj" in name or "v_proj" in name:
                self.base_weights[name] = param.data.clone()

    def load(self, name, path):
        if not os.path.exists(path):
            return False
        data = torch.load(path, map_location="cpu", weights_only=False)
        weights = data.get("lora_state_dict", data)

        pairs = {}
        for k, v in weights.items():
            if ".lora_A" in k:
                base = k.replace(".lora_A", "")
                b = k.replace(".lora_A", ".lora_B")
                if b in weights:
                    pairs[base] = {"A": v, "B": weights[b]}

        if pairs:
            self.adapters[name] = pairs
            return True
        return False

    def apply(self, name, alpha=8, rank=4):
        if name not in self.adapters:
            return False

        with torch.no_grad():
            for lora_key, lora in self.adapters[name].items():
                for pname, param in self.model.named_parameters():
                    if self._match(lora_key, pname):
                        if lora["A"].shape[1] == param.shape[1]:
                            delta = (lora["B"].to(device) @ lora["A"].to(device)) * (
                                alpha / rank
                            )
                            param.data.add_(delta.to(param.dtype))
                        break
        return True

    def _match(self, lora, model):
        lora_parts = set(re.findall(r"\d+|[a-z]+", lora.lower()))
        model_parts = set(re.findall(r"\d+|[a-z]+", model.lower()))

        lora_sig = ".".join(sorted(lora_parts))
        model_sig = ".".join(sorted(model_parts))

        return lora_sig == model_sig

    def restore(self):
        with torch.no_grad():
            for name, param in self.model.named_parameters():
                if name in self.base_weights:
                    param.data.copy_(self.base_weights[name])


class Router:
    def __init__(self):
        self.patterns = {
            "math": [
                (r"\b\d+\s*[\+\-\*\/\=\^]\s*\d+", 3),
                (r"\bsolve\b", 2),
                (r"\bcalculate\b", 2),
                (r"\bcompute\b", 2),
                (r"\bwhat\s+is\s+\d+", 3),
                (r"\bequation\b", 1),
                (r"\bsum\b", 1),
                (r"\bpercentage\b", 1),
                (r"\bratio\b", 1),
                (r"\bfactor\b", 1),
            ],
            "code": [
                (r"\bdef\s+\w+\s*\(", 3),
                (r"\bclass\s+\w+", 3),
                (r"\bimport\s+\w+", 2),
                (r"\bfunction\b", 1),
                (r"\breturn\b", 1),
                (r"\bpython\b", 1),
                (r"\bcode\b", 1),
                (r"\bprogram\b", 1),
                (r"\balgorithm\b", 1),
                (r"\bwrite\s+(a)?\s*(function|program|code)", 2),
            ],
            "knowledge": [
                (r"^what\s+is\b", 2),
                (r"^who\s+is\b", 2),
                (r"\bexplain\b", 1),
                (r"\bdefine\b", 1),
                (r"\bhistory\b", 1),
                (r"\btheory\b", 1),
                (r"\bscience\b", 1),
                (r"\bdescribe\b", 1),
                (r"\btell\s+me\b", 1),
            ],
            "reasoning": [
                (r"\bwhy\b", 2),
                (r"\bhow\s+do\b", 2),
                (r"\bcompare\b", 2),
                (r"\banalyze\b", 2),
                (r"\bevaluate\b", 1),
                (r"\bconsider\b", 1),
            ],
            "conversation": [
                (r"^hello\b", 3),
                (r"^hi\b", 3),
                (r"^hey\b", 3),
                (r"\bhow\s+are\s+you\b", 2),
                (r"\bthank\b", 1),
                (r"\bhelp\s+me\b", 1),
            ],
        }

    def route(self, text):
        scores = {}
        for expert, pats in self.patterns.items():
            score = 0
            for pattern, weight in pats:
                matches = len(re.findall(pattern, text, re.I))
                score += matches * weight
            scores[expert] = score

        if max(scores.values()) == 0:
            return "reasoning"
        return max(scores, key=scores.get)


class Agent:
    def __init__(self):
        print("Loading MoE Agent...")

        self.tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL, torch_dtype=torch.float32
        )
        self.model = self.model.to(device)

        self.lora = LoRAManager(self.model)
        for exp in EXPERTS:
            self.lora.load(exp, f"{ADAPTERS_DIR}/{exp}_lora_adapter.pt")

        self.router = Router()
        print("Ready!\n")

    def generate(self, prompt, max_tokens=100):
        inputs = self.tokenizer(prompt, return_tensors="pt")
        ids = inputs["input_ids"].to(device)

        with torch.no_grad():
            out = self.model.generate(
                ids,
                max_new_tokens=max_tokens,
                temperature=0.7,
                do_sample=True,
                top_k=40,
                pad_token_id=self.tokenizer.pad_token_id,
            )
        return self.tokenizer.decode(out[0], skip_special_tokens=True)

    def run(self, query):
        expert = self.router.route(query)

        self.lora.restore()
        self.lora.apply(expert)

        prompt = f"{query}\n\nAnswer:"
        result = self.generate(prompt, max_tokens=60)

        answer = result.split("Answer:")[-1].strip() if "Answer:" in result else result
        print(f"[{expert.upper()}] {answer}")
        return answer

    def chat(self):
        print("MoE Agent (type 'quit' to exit)\n")

        while True:
            try:
                q = input("You: ").strip()
                if q.lower() in ["quit", "exit", "bye"]:
                    break
                if q:
                    self.run(q)
            except KeyboardInterrupt:
                break

        print("\nBye!")


if __name__ == "__main__":
    import sys

    agent = Agent()

    if len(sys.argv) > 1 and sys.argv[1] == "--chat":
        agent.chat()
    else:
        print("\nRunning test queries...\n")
        sys.stdout.flush()
        agent.run("What is 7 + 8?")
        sys.stdout.flush()
        agent.run("Write a Python hello world")
        sys.stdout.flush()
        agent.run("Why is the sky blue?")
        sys.stdout.flush()
        print("\nDone!")
