"""
MoE System with Real LoRA Integration
Applies expert LoRA adapters during generation based on routing
"""

import os
import re
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

try:
    import torch_directml

    device = torch_directml.device()
except:
    device = torch.device("cpu")

BASE_MODEL = "M:/training_models/gemma-3-270m-it"
EXPERTS = ["math", "reasoning", "code", "conversation", "knowledge"]
ADAPTERS_DIR = "M:/tinyllama/experts"


class LoRAManager:
    """Manages LoRA adapter loading and application"""

    def __init__(self, model):
        self.model = model
        self.adapters = {}
        self.base_weights = {}
        self._save_base_weights()

    def _save_base_weights(self):
        """Save original model weights for target modules"""
        target_modules = ["q_proj", "v_proj", "k_proj", "o_proj"]

        for name, param in self.model.named_parameters():
            for target in target_modules:
                if target in name and "weight" in name:
                    self.base_weights[name] = param.data.clone()

        print(f"  Saved {len(self.base_weights)} base weight tensors")

    def load_adapter(self, name, path):
        """Load a LoRA adapter from file"""
        if not os.path.exists(path):
            print(f"  Adapter not found: {path}")
            return False

        data = torch.load(path, map_location="cpu", weights_only=False)

        if "lora_state_dict" in data:
            adapter_weights = data["lora_state_dict"]
        else:
            adapter_weights = data

        lora_pairs = {}
        for key, value in adapter_weights.items():
            if ".lora_A" in key:
                base_key = key.replace(".lora_A", "")
                lora_b_key = key.replace(".lora_A", ".lora_B")
                if lora_b_key in adapter_weights:
                    lora_pairs[base_key] = {
                        "A": value,
                        "B": adapter_weights[lora_b_key],
                    }

        if lora_pairs:
            self.adapters[name] = lora_pairs
            print(f"  Loaded LoRA '{name}': {len(lora_pairs)} layer pairs")
            return True
        else:
            print(f"  No valid LoRA pairs in {path}")
            return False

    def _convert_lora_key_to_model_key(self, lora_key):
        """Convert lora key to model parameter key"""
        model_key = lora_key.replace("_", ".")

        parts = model_key.split(".")
        for i, p in enumerate(parts):
            if p.isdigit():
                parts[i - 1] = parts[i - 1] + "." + p
                parts[i] = ""

        model_key = ".".join(p for p in parts if p)
        model_key = model_key.replace(".self.attn.", ".self_attn.")

        return model_key + ".weight"

    def apply_adapter(self, adapter_name, alpha=8, rank=4):
        """Apply LoRA adapter to model"""
        if adapter_name not in self.adapters:
            print(f"  Adapter '{adapter_name}' not loaded")
            return False

        adapter = self.adapters[adapter_name]
        scaling = alpha / rank
        applied = 0

        with torch.no_grad():
            for lora_key, lora_data in adapter.items():
                lora_a = lora_data["A"].to(device)
                lora_b = lora_data["B"].to(device)

                model_key = self._convert_lora_key_to_model_key(lora_key)

                for name, param in self.model.named_parameters():
                    if self._keys_match(model_key, name):
                        delta = (lora_b @ lora_a) * scaling
                        param.data.add_(delta.to(param.dtype))
                        applied += 1
                        break

        return applied > 0

    def _keys_match(self, lora_key, model_key):
        """Check if LoRA key corresponds to model key"""
        lora_parts = re.findall(r"\d+|[a-z]+", lora_key.lower())
        model_parts = re.findall(r"\d+|[a-z]+", model_key.lower())

        lora_sig = ".".join(lora_parts[-5:])
        model_sig = ".".join(model_parts[-5:])

        return lora_sig == model_sig

    def restore_base(self):
        """Restore original model weights"""
        with torch.no_grad():
            for name, param in self.model.named_parameters():
                if name in self.base_weights:
                    param.data.copy_(self.base_weights[name])


class FastRouter(nn.Module):
    def __init__(self, vocab_size, embed_dim=128, hidden_dim=256, num_classes=5):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.conv1 = nn.Conv1d(embed_dim, hidden_dim, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1)
        self.pool = nn.AdaptiveMaxPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim // 2, num_classes),
        )

    def forward(self, x):
        x = self.embedding(x)
        x = x.permute(0, 2, 1)
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = self.pool(x).squeeze(-1)
        return self.fc(x)


class TextVectorizer:
    def __init__(self, vocab, max_len=128):
        self.vocab = vocab
        self.max_len = max_len

    def transform(self, text):
        tokens = re.findall(r"\b\w+\b", text.lower())
        indices = [self.vocab.get(t, 1) for t in tokens[: self.max_len]]
        indices = indices + [0] * (self.max_len - len(indices))
        return torch.tensor([indices], dtype=torch.long)


class HybridRouter:
    def __init__(self, neural_router, vectorizer):
        self.neural_router = neural_router
        self.vectorizer = vectorizer

        self.patterns = {
            "math": [
                r"\b\d+\s*[\+\-\*\/\=]\s*\d+",
                r"\bsolve\b",
                r"\bcalculate\b",
                r"\bequation\b",
                r"\bsum\b",
                r"\bcompute\b",
                r"\bwhat\s+is\s+\d+",
                r"\bfactor\b",
                r"\bprobability\b",
                r"\bpercentage\b",
            ],
            "code": [
                r"\bdef\s+\w+",
                r"\bclass\s+\w+",
                r"\bimport\s+\w+",
                r"\bfunction\b",
                r"\breturn\b",
                r"\bpython\b",
                r"\bcode\b",
            ],
            "knowledge": [
                r"^what\s+is\s+(the|a)\b",
                r"^who\s+is\b",
                r"\bexplain\b",
                r"\bdefine\b",
                r"\bhistory\b",
                r"\btheory\b",
                r"\bscience\b",
                r"^why\s+is\s+(the|sky|earth|sun)\b",
            ],
            "reasoning": [
                r"^why\b",
                r"^how\s+do\b",
                r"\bcompare\b",
                r"\banalyze\b",
                r"\bbecause\b",
                r"\btherefore\b",
            ],
            "conversation": [
                r"^hello\b",
                r"^hi\b",
                r"\bhow\s+are\b",
                r"\bthank\b",
                r"\bhelp\s+me\b",
                r"\badvice\b",
            ],
        }

        self.compiled = {
            e: [re.compile(p, re.I) for p in pats] for e, pats in self.patterns.items()
        }

    def route(self, text):
        scores = {
            e: sum(len(p.findall(text)) for p in pats)
            for e, pats in self.compiled.items()
        }

        if self.neural_router:
            with torch.no_grad():
                x = self.vectorizer.transform(text)
                logits = self.neural_router(x)
                probs = F.softmax(logits, dim=-1)[0]
                labels = ["math", "reasoning", "code", "conversation", "knowledge"]
                for i, label in enumerate(labels):
                    if scores[label] >= 2:
                        scores[label] = scores[label] * 0.7 + probs[i].item() * 0.3
                    else:
                        scores[label] = (
                            scores.get(label, 0) * 0.3 + probs[i].item() * 0.7
                        )

        best = max(scores, key=scores.get)
        total = sum(max(s, 0) for s in scores.values()) or 1
        return best, scores, min(scores[best] / total, 1.0)


class MoEWithLoRA:
    def __init__(self):
        print("=" * 60)
        print("MoE WITH LoRA INTEGRATION")
        print("=" * 60)
        print(f"Device: {device}")

        print("\n[1/3] Loading base model...")
        self.tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL, torch_dtype=torch.float32
        )
        self.model = self.model.to(device)

        print("\n[2/3] Loading LoRA adapters...")
        self.lora_manager = LoRAManager(self.model)
        for expert in EXPERTS:
            path = f"{ADAPTERS_DIR}/{expert}_lora_adapter.pt"
            self.lora_manager.load_adapter(expert, path)

        print("\n[3/3] Loading router...")
        router_path = f"{ADAPTERS_DIR}/lightweight_router.pt"
        if os.path.exists(router_path):
            ckpt = torch.load(router_path, map_location="cpu", weights_only=False)
            cfg = ckpt["config"]
            self.neural_router = FastRouter(
                cfg["vocab_size"],
                cfg["embed_dim"],
                cfg["hidden_dim"],
                cfg["num_classes"],
            )
            self.neural_router.load_state_dict(ckpt["model_state"])
            self.neural_router.eval()
            self.router = HybridRouter(
                self.neural_router, TextVectorizer(ckpt["vocab"])
            )
            print("  Neural router loaded")
        else:
            self.router = HybridRouter(None, None)
            print("  Pattern-only routing")

        self.current_expert = None
        print("\n" + "=" * 60)
        print("READY")
        print("=" * 60)

    def generate(self, prompt, max_new_tokens=80, use_lora=True):
        expert, scores, conf = self.router.route(prompt)
        print(f"\n[Routing: {expert} ({conf * 100:.0f}%)]")

        if use_lora and expert in self.lora_manager.adapters:
            self.lora_manager.restore_base()
            self.lora_manager.apply_adapter(expert)
            self.current_expert = expert

        inputs = self.tokenizer(prompt, return_tensors="pt")
        input_ids = inputs["input_ids"].to(device)

        with torch.no_grad():
            output = self.model.generate(
                input_ids,
                max_new_tokens=max_new_tokens,
                temperature=0.7,
                do_sample=True,
                top_k=40,
                pad_token_id=self.tokenizer.pad_token_id,
            )

        return self.tokenizer.decode(output[0], skip_special_tokens=True)

    def chat(self):
        print("\n" + "=" * 60)
        print("MoE CHAT (quit to exit)")
        print("=" * 60)
        while True:
            try:
                user = input("\nYou: ").strip()
                if user.lower() in ["quit", "exit", "bye"]:
                    break
                if user:
                    print(self.generate(user))
            except KeyboardInterrupt:
                break
        print("\nGoodbye!")

    def test(self):
        print("\n" + "=" * 60)
        print("TESTING")
        print("=" * 60)
        tests = [
            ("What is 5 + 7?", "math"),
            ("def add(a, b):", "code"),
            ("Why is the sky blue?", "knowledge"),
            ("Hello there!", "conversation"),
        ]
        for prompt, expected in tests:
            expert, _, conf = self.router.route(prompt)
            print(f"\nQ: {prompt}")
            print(
                f"  Route: {expert} (expected: {expected}) {'OK' if expert == expected else 'WRONG'}"
            )
            resp = self.generate(prompt, max_new_tokens=50)
            print(f"  Response: {resp[:120]}...")


if __name__ == "__main__":
    import sys

    moe = MoEWithLoRA()
    if len(sys.argv) > 1 and sys.argv[1] == "--chat":
        moe.chat()
    else:
        moe.test()
