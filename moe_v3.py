import os
import re
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
import json

try:
    import torch_directml

    device = torch_directml.device()
except:
    device = torch.device("cpu")

BASE_MODEL = "M:/training_models/gemma-3-270m-it"
EXPERTS = ["math", "reasoning", "code", "conversation", "knowledge"]
LABEL2ID = {label: i for i, label in enumerate(EXPERTS)}
ID2LABEL = {i: label for i, label in enumerate(EXPERTS)}


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
        x = self.fc(x)
        return x


class TextVectorizer:
    def __init__(self, vocab, max_len=128):
        self.vocab = vocab
        self.max_len = max_len

    def tokenize(self, text):
        text = text.lower()
        tokens = re.findall(r"\b\w+\b", text)
        return tokens

    def transform(self, text):
        tokens = self.tokenize(text)
        indices = [self.vocab.get(t, 1) for t in tokens[: self.max_len]]
        indices = indices + [0] * (self.max_len - len(indices))
        return torch.tensor([indices], dtype=torch.long)


class HybridRouter:
    def __init__(self, neural_router, vectorizer):
        self.neural_router = neural_router
        self.vectorizer = vectorizer

        self.strong_patterns = {
            "code": [
                r"\bdef\s+\w+\s*\(",
                r"\bclass\s+\w+\s*[:\(]",
                r"\bimport\s+\w+",
                r"\bfunction\b.*\(",
                r"\breturn\s+\w",
                r"\bprint\s*\(",
                r"\bfor\s+\w+\s+in\s+",
                r"\bwhile\s+\w",
                r"\bif\s+\w+\s*:",
                r"\belse\s*:",
                r"\belif\s*:",
                r"\btry\s*:",
                r"\bexcept\s*:",
                r"```python",
                r"```javascript",
                r"```java",
                r"```cpp",
                r"\bwrite\s+(a|the)?\s*(python|javascript|java|code|function|program)",
                r"\bimplement\s+(a|the)?\s*(function|class|algorithm)",
                r"\bdebug\s+",
                r"\bapi\b",
                r"\bvariable\b",
                r"\barray\b",
                r"\bsort\s+(a|the)?\s*list",
                r"\bsort\s+list\b",
            ],
            "knowledge": [
                r"^what\s+is\b",
                r"^what\s+are\b",
                r"^who\s+is\b",
                r"^who\s+was\b",
                r"^when\s+did\b",
                r"^where\s+is\b",
                r"\bdefine\s+\w+$",
                r"\bdefinition\s+of\b",
                r"\bhistory\s+of\b",
                r"\btell\s+me\s+about\b",
                r"\bexplain\s+(the|what|how|a|theory)\b",
                r"\bdescribe\s+(the|a|how)\b",
                r"\bscience\s+of\b",
                r"\btechnology\s+of\b",
                r"\bwhy\s+is\s+the\b",
                r"\btheory\s+of\b",
                r"\brelativity\b",
                r"\bquantum\b",
                r"^explain\b",
            ],
            "reasoning": [
                r"^why\b",
                r"^how\s+do\b",
                r"\banalyze\s+",
                r"\bcompare\s+",
                r"\bcontrast\b",
                r"\bevaluate\b",
                r"\bthink\s+about\b",
                r"\bconsider\s+(the|how|why)\b",
                r"\bbecause\b",
                r"\btherefore\b",
                r"\bhowever\b",
                r"\bwhat\s+if\b",
                r"\bscenario\b",
                r"\bimplication\b",
                r"\bconsequence\b",
                r"\breasoning\b",
            ],
            "math": [
                r"\b\d+\s*[\+\-\*\/\=\^]\s*\d+",
                r"\bsolve\s+(for|x|the)",
                r"\bcalculate\b",
                r"\bcompute\b",
                r"\bequation\b",
                r"\bfind\s+(the\s+)?(x|y|value|sum|product|difference)",
                r"\bsum\s+of\b",
                r"\bproduct\s+of\b",
                r"\bfactor\b",
                r"\bprobability\b",
                r"\bintegral\b",
                r"\bderivative\b",
                r"\bsquare\s+root\b",
                r"\bpercentage\b",
                r"\bthe\s+ratio\b",
                r"\bratio\s+of\b",
                r"\bwhat\s+is\s+\d+\s*[\+\-\*\/]",
            ],
            "conversation": [
                r"^hello\b",
                r"^hi\s",
                r"^hey\b",
                r"\bhow\s+are\s+you\b",
                r"\bthank\s*you\b",
                r"\bplease\s+help\b",
                r"\bsorry\b",
                r"\bi\s+need\s+advice\b",
                r"\bcan\s+you\s+help\b",
                r"\bcould\s+you\b",
                r"\bi\s+feel\b",
                r"\badvice\s+on\b",
                r"\bsuggest\b",
                r"\bgood\s+morning\b",
                r"\bgood\s+evening\b",
                r"\bnice\s+to\s+meet\b",
            ],
        }

        self.compiled = {}
        for expert, patterns in self.strong_patterns.items():
            self.compiled[expert] = [re.compile(p, re.IGNORECASE) for p in patterns]

    def pattern_scores(self, text):
        scores = {}
        for expert, patterns in self.compiled.items():
            count = 0
            for p in patterns:
                count += len(p.findall(text))
            scores[expert] = count
        return scores

    def route(self, text):
        pattern_scores = self.pattern_scores(text)

        if self.neural_router is not None:
            with torch.no_grad():
                x = self.vectorizer.transform(text)
                logits = self.neural_router(x)
                neural_probs = F.softmax(logits, dim=-1)[0]
                neural_scores = {
                    ID2LABEL[i]: neural_probs[i].item() for i in range(len(EXPERTS))
                }
        else:
            neural_scores = {e: 0.2 for e in EXPERTS}

        max_pattern = max(pattern_scores.values()) if pattern_scores else 0

        if max_pattern >= 1:
            best_pattern = max(pattern_scores, key=pattern_scores.get)
            if pattern_scores[best_pattern] >= 2:
                return best_pattern, pattern_scores, 0.9

        combined = {}
        for expert in EXPERTS:
            ps = pattern_scores.get(expert, 0)
            ns = neural_scores.get(expert, 0.2)

            if ps >= 1:
                combined[expert] = ps * 0.6 + ns * 0.4
            else:
                combined[expert] = ns

        best = max(combined, key=combined.get)
        confidence = combined[best]

        return best, combined, confidence


class MoESystem:
    def __init__(self):
        print("=" * 60)
        print("LOADING MoE SYSTEM v3 (HYBRID ROUTER)")
        print("=" * 60)
        print(f"Device: {device}")

        print("\nLoading base model...")
        self.tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
        self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL, torch_dtype=torch.float32
        )
        self.model = self.model.to(device)

        print("\nLoading hybrid router...")
        router_path = "M:/tinyllama/experts/lightweight_router.pt"
        if os.path.exists(router_path):
            checkpoint = torch.load(router_path, map_location="cpu", weights_only=False)
            config = checkpoint["config"]
            neural_router = FastRouter(
                vocab_size=config["vocab_size"],
                embed_dim=config["embed_dim"],
                hidden_dim=config["hidden_dim"],
                num_classes=config["num_classes"],
            )
            neural_router.load_state_dict(checkpoint["model_state"])
            neural_router.eval()
            vectorizer = TextVectorizer(checkpoint["vocab"])
            self.router = HybridRouter(neural_router, vectorizer)
            print(f"  Loaded neural router + pattern fallback")
        else:
            self.router = HybridRouter(None, None)
            print("  Using pattern-only routing")

        print("\nLoading LoRA adapters...")
        self.lora_adapters = {}
        for name in EXPERTS:
            path = f"M:/tinyllama/experts/{name}_lora_adapter.pt"
            if os.path.exists(path):
                data = torch.load(path, map_location="cpu", weights_only=False)
                self.lora_adapters[name] = data
                print(f"  Loaded: {name}")

        print("\n" + "=" * 60)
        print("MoE SYSTEM READY")
        print("=" * 60)

    def generate(self, prompt, max_new_tokens=150, temperature=0.7, top_k=40):
        expert, probs, confidence = self.router.route(prompt)

        total = sum(probs.values()) or 1
        normalized = {e: p / total for e, p in probs.items()}

        inputs = self.tokenizer(prompt, return_tensors="pt")
        input_ids = inputs["input_ids"].to(device)

        top3 = sorted(normalized.items(), key=lambda x: -x[1])[:3]
        print(f"\n[Routing: {expert} ({confidence * 100:.1f}%)]")
        print(f"[Top 3: " + ", ".join(f"{e}:{p * 100:.0f}%" for e, p in top3) + "]")

        with torch.no_grad():
            output = self.model.generate(
                input_ids,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=True,
                top_k=top_k,
                pad_token_id=self.tokenizer.pad_token_id,
            )

        return self.tokenizer.decode(output[0], skip_special_tokens=True)

    def chat(self):
        print("\n" + "=" * 60)
        print("MoE CHAT v3 - Hybrid Router (type 'quit' to exit)")
        print("=" * 60)

        while True:
            try:
                user = input("\nYou: ").strip()
                if user.lower() in ["quit", "exit", "bye"]:
                    break
                if not user:
                    continue

                response = self.generate(user)
                print(f"\nMoE: {response}")
            except KeyboardInterrupt:
                break

        print("\nGoodbye!")

    def test(self):
        print("\n" + "=" * 60)
        print("TESTING MoE SYSTEM v3 (HYBRID)")
        print("=" * 60)

        tests = [
            ("Solve: 2 + 2 = ?", "math"),
            ("def hello_world():", "code"),
            ("Why is the sky blue?", "knowledge"),
            ("Hello, how are you?", "conversation"),
            ("What is quantum computing?", "knowledge"),
            ("Write a Python function to sort a list", "code"),
            ("Calculate 15% of 200", "math"),
            ("Explain the theory of relativity", "knowledge"),
            ("Compare Python and JavaScript", "reasoning"),
            ("I need advice on learning programming", "conversation"),
        ]

        correct = 0
        for prompt, expected in tests:
            expert, probs, conf = self.router.route(prompt)
            status = "OK" if expert == expected else "WRONG"
            if expert == expected:
                correct += 1
            print(f"\nQ: {prompt}")
            print(
                f"  Expected: {expected}, Got: {expert} ({conf * 100:.1f}%) [{status}]"
            )

        print(f"\n" + "=" * 60)
        print(
            f"ROUTER ACCURACY: {correct}/{len(tests)} ({100 * correct / len(tests):.0f}%)"
        )
        print("=" * 60)


if __name__ == "__main__":
    import sys

    moe = MoESystem()

    if len(sys.argv) > 1:
        if sys.argv[1] == "--test":
            moe.test()
        elif sys.argv[1] == "--chat":
            moe.chat()
        else:
            prompt = " ".join(sys.argv[1:])
            print(moe.generate(prompt))
    else:
        moe.test()
        print("\n--- Starting interactive chat ---")
        moe.chat()
