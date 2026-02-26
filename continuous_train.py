"""
Continuous MoE Training Pipeline
Loop: Download -> Clean -> Train LoRAs -> Train Router -> Evaluate -> Repeat
"""

import os
import sys
import json
import re
import time
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
)
from datasets import load_dataset
from collections import defaultdict
from sklearn.metrics import accuracy_score
import numpy as np

random.seed(42)
torch.manual_seed(42)

# Config
BASE_MODEL = "M:/training_models/gemma-3-270m-it"
DATA_DIR = "M:/tinyllama/cleaned_data"
EXPERTS_DIR = "M:/tinyllama/experts"
EXPERTS = [
    "math",
    "code",
    "reasoning",
    "conversation",
    "knowledge",
    "science",
    "writing",
    "creative",
]

# Data sources per domain
DATA_SOURCES = {
    "math": [
        ("gsm8k", "main", "question", "answer"),
        ("MuskPenguin/math_qa_alpaca", None, "instruction", "output"),
        ("lighteval/MATH", "all", "problem", "solution"),
    ],
    "code": [
        ("m-a-p/CodeFeedback-Filtered-Instruction", None, "query", "answer"),
        ("HuggingFaceH4/CodeAlpaca_20K", None, "prompt", "completion"),
        ("sahil2801/CodeAlpaca-120k", None, "instruction", "output"),
    ],
    "reasoning": [
        ("tasksource/logiqa", None, "context", "answer"),
        ("openbookqa", "main", "question_stem", "choices"),
    ],
    "conversation": [
        ("Aeala/ShareGPT_Vicuna_unfiltered", None, "conversations", None),
        ("OpenAssistant/oasst1", None, "text", None),
    ],
    "knowledge": [
        ("garage-bAInd/Open-Platypus", None, "instruction", "output"),
        ("databricks/databricks-dolly-15k", None, "instruction", "response"),
        ("tomg-group-umd/crazy_qa", None, "question", "answer"),
    ],
    "science": [
        ("allenai/sciq", None, "question", "correct_answer"),
        ("wiki_qa", None, "question", "answer"),
    ],
    "writing": [
        ("HuggingFaceH4/instruction-dataset", None, "prompt", "completion"),
    ],
    "creative": [
        ("Gustavosta/Stories-Dataset", None, "text", None),
    ],
}


def clean_text(text):
    if not text:
        return ""
    text = str(text)
    text = re.sub(r"<[^>]+>", "", text)
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[\x00-\x1f\x7f-\x9f]", "", text)
    return text.strip()


def categorize_text(text):
    text_lower = text.lower()

    patterns = {
        "math": [
            (r"\b\d+\s*[\+\-\*\/\=\^]\s*\d+", 3),
            (r"\bsolve\b", 2),
            (r"\bcalculate\b", 2),
            (r"\bequation\b", 1),
            (r"\bsum\b", 1),
            (r"\bfactor\b", 1),
        ],
        "code": [
            (r"\bdef\s+\w+\s*\(", 3),
            (r"\bclass\s+\w+", 3),
            (r"\bimport\s+\w+", 2),
            (r"\bfunction\b", 1),
            (r"\bpython\b", 1),
            (r"\balgorithm\b", 1),
        ],
        "reasoning": [
            (r"\bwhy\b", 2),
            (r"\bhow\b", 1),
            (r"\bcompare\b", 2),
            (r"\banalyze\b", 2),
            (r"\bbecause\b", 1),
            (r"\btherefore\b", 1),
        ],
        "conversation": [
            (r"^hello\b", 3),
            (r"^hi\b", 3),
            (r"\bhow\s+are\b", 2),
            (r"\bthank\b", 1),
            (r"\bhelp\s+me\b", 1),
        ],
        "knowledge": [
            (r"^what\s+is\b", 2),
            (r"^who\s+is\b", 2),
            (r"\bexplain\b", 1),
            (r"\bdefine\b", 1),
            (r"\bhistory\b", 1),
        ],
        "science": [
            (r"\bscience\b", 2),
            (r"\bphysics\b", 2),
            (r"\bchemistry\b", 2),
            (r"\bbiology\b", 2),
            (r"\bexperiment\b", 1),
        ],
        "writing": [
            (r"\bwrite\b", 1),
            (r"\bessay\b", 2),
            (r"\bparagraph\b", 1),
            (r"\bstory\b", 1),
            (r"\barticle\b", 1),
        ],
        "creative": [
            (r"\bcreative\b", 2),
            (r"\bimagine\b", 1),
            (r"\bfiction\b", 2),
            (r"\bpoem\b", 2),
            (r"\btale\b", 1),
        ],
    }

    scores = {}
    for domain, pats in patterns.items():
        score = sum(len(re.findall(p, text_lower)) * w for p, w in pats)
        scores[domain] = score

    if max(scores.values()) == 0:
        return None
    return max(scores, key=scores.get)


def format_sample(instruction, output=None):
    instruction = clean_text(instruction)
    if not instruction or len(instruction) < 15:
        return None
    if output:
        output = clean_text(output)
        if len(output) < 5:
            output = None
    if output:
        return f"Instruction: {instruction}\nResponse: {output}"
    return f"Instruction: {instruction}"


def download_data(domain, max_samples=800):
    """Download data for a domain"""
    samples = []
    sources = DATA_SOURCES.get(domain, [])

    for source_info in sources:
        if len(samples) >= max_samples:
            break

        name, config, field_in, field_out = source_info
        print(f"    Fetching {name}...")

        try:
            if config:
                ds = load_dataset(name, config, split="train", trust_remote_code=True)
            else:
                ds = load_dataset(name, split="train", trust_remote_code=True)

            for item in ds:
                if len(samples) >= max_samples:
                    break

                try:
                    if field_in == "conversations":
                        convs = item.get("conversations", [])
                        if isinstance(convs, list):
                            for msg in convs:
                                if isinstance(msg, dict):
                                    role = msg.get("from", msg.get("role", ""))
                                    content = msg.get("value", msg.get("content", ""))
                                    if role in ["human", "user"] and content:
                                        sample = format_sample(content)
                                        if sample:
                                            samples.append(sample)
                                            break
                    elif field_in == "text":
                        text = item.get("text", "")
                        if text:
                            sample = format_sample(text)
                            if sample:
                                samples.append(sample)
                    elif field_in == "choices":
                        q = item.get("question_stem", "")
                        choices = item.get("choices", {})
                        if isinstance(choices, dict):
                            texts = choices.get("text", [])
                            if q and texts:
                                sample = format_sample(
                                    f"{q} Options: {', '.join(texts)}"
                                )
                                if sample:
                                    samples.append(sample)
                    else:
                        instruction = (
                            item.get(field_in, "")
                            or item.get("instruction", "")
                            or item.get("prompt", "")
                        )
                        output = (
                            item.get(field_out, "")
                            or item.get("output", "")
                            or item.get("answer", "")
                            or item.get("completion", "")
                        )

                        if instruction:
                            sample = format_sample(
                                instruction, output if output else None
                            )
                            if sample:
                                samples.append(sample)
                except:
                    continue
        except Exception as e:
            print(f"      Error: {str(e)[:50]}")

    return samples


def prepare_data():
    """Download and prepare fresh data"""
    print("\n" + "=" * 60)
    print("PHASE 1: DATA PREPARATION")
    print("=" * 60)

    all_samples = defaultdict(list)

    for domain in EXPERTS:
        print(f"\n[{domain.upper()}]")
        new_samples = download_data(domain, max_samples=600)

        for sample in new_samples:
            category = categorize_text(sample)
            if category == domain:
                if len(sample) > 40 and len(sample) < 3000:
                    all_samples[domain].append(sample)
            elif category is None:
                all_samples[domain].append(sample)

        print(f"    Collected: {len(all_samples[domain])} samples")

    print("\nMerging with existing data...")
    os.makedirs(DATA_DIR, exist_ok=True)

    stats = {}
    for domain in EXPERTS:
        existing_path = f"{DATA_DIR}/{domain}_train.json"
        existing = []
        if os.path.exists(existing_path):
            with open(existing_path) as f:
                existing = json.load(f)

        combined = list(set(existing + all_samples[domain]))
        random.shuffle(combined)

        n = len(combined)
        n_train = int(n * 0.8)
        n_val = int(n * 0.1)

        with open(f"{DATA_DIR}/{domain}_train.json", "w") as f:
            json.dump(combined[:n_train], f)
        with open(f"{DATA_DIR}/{domain}_val.json", "w") as f:
            json.dump(combined[n_train : n_train + n_val], f)
        with open(f"{DATA_DIR}/{domain}_test.json", "w") as f:
            json.dump(combined[n_train + n_val :], f)

        stats[domain] = n
        print(f"  {domain}: {len(existing)} + {len(all_samples[domain])} -> {n}")

    return stats


class LoRADataset(Dataset):
    def __init__(self, data, tokenizer, max_len=256):
        self.data = data
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        text = self.data[idx]
        enc = self.tokenizer(
            text,
            max_length=self.max_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        ids = enc["input_ids"].squeeze()
        mask = enc["attention_mask"].squeeze()
        labels = ids.clone()
        labels[labels == self.tokenizer.pad_token_id] = -100
        return {"input_ids": ids, "attention_mask": mask, "labels": labels}


class LoRALayer(nn.Module):
    def __init__(self, original, rank=8, alpha=16):
        super().__init__()
        self.original = original
        self.rank = rank
        self.alpha = alpha

        self.A = nn.Parameter(torch.zeros(rank, original.in_features))
        self.B = nn.Parameter(torch.zeros(original.out_features, rank))

        nn.init.kaiming_uniform_(self.A, a=5**0.5)
        nn.init.zeros_(self.B)

        for p in self.original.parameters():
            p.requires_grad = False

    def forward(self, x):
        return self.original(x) + (self.alpha / self.rank) * (x @ self.A.T @ self.B.T)


def add_lora(model, rank=8, alpha=16, targets=["q_proj", "v_proj", "k_proj", "o_proj"]):
    lora_layers = []
    for name, module in model.named_modules():
        if any(t in name for t in targets) and isinstance(module, nn.Linear):
            parts = name.rsplit(".", 1)
            if len(parts) == 2:
                parent_name, child_name = parts
                parent = model
                for p in parent_name.split("."):
                    if p:
                        parent = getattr(parent, p)
                lora = LoRALayer(module, rank, alpha)
                setattr(parent, child_name, lora)
                lora_layers.append((name, lora))
    return lora_layers


def train_lora(domain, epochs=2, batch_size=2):
    """Train LoRA adapter for a domain"""
    print(f"\n  Training {domain}...")

    device = torch.device("cpu")

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(BASE_MODEL, torch_dtype=torch.float32)
    model = model.to(device)

    lora_layers = add_lora(model)
    print(f"    LoRA layers: {len(lora_layers)}")

    train_path = f"{DATA_DIR}/{domain}_train.json"
    val_path = f"{DATA_DIR}/{domain}_val.json"

    if not os.path.exists(train_path):
        print(f"    No training data found")
        return None

    with open(train_path) as f:
        train_data = json.load(f)[:500]
    with open(val_path) as f:
        val_data = json.load(f)[:100]

    if len(train_data) < 20:
        print(f"    Not enough data: {len(train_data)}")
        return None

    train_ds = LoRADataset(train_data, tokenizer)
    val_ds = LoRADataset(val_data, tokenizer)

    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_dl = DataLoader(val_ds, batch_size=batch_size)

    params = []
    for _, layer in lora_layers:
        params.extend([layer.A, layer.B])

    optimizer = torch.optim.AdamW(params, lr=1e-4)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, len(train_dl), len(train_dl) * epochs
    )

    best_loss = float("inf")

    for epoch in range(epochs):
        model.train()
        total_loss = 0

        for batch in train_dl:
            ids = batch["input_ids"].to(device)
            mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            optimizer.zero_grad()
            out = model(input_ids=ids, attention_mask=mask, labels=labels)
            out.loss.backward()
            optimizer.step()
            scheduler.step()

            total_loss += out.loss.item()

        avg_loss = total_loss / len(train_dl)

        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in val_dl:
                ids = batch["input_ids"].to(device)
                mask = batch["attention_mask"].to(device)
                labels = batch["labels"].to(device)
                out = model(input_ids=ids, attention_mask=mask, labels=labels)
                val_loss += out.loss.item()

        avg_val = val_loss / len(val_dl)
        print(f"    Epoch {epoch + 1}: train={avg_loss:.4f}, val={avg_val:.4f}")

        if avg_val < best_loss:
            best_loss = avg_val
            state = {}
            for name, layer in lora_layers:
                state[f"{name}.lora_A"] = layer.A.data.cpu()
                state[f"{name}.lora_B"] = layer.B.data.cpu()

            save_path = f"{EXPERTS_DIR}/{domain}_lora_adapter.pt"
            torch.save({"lora_state_dict": state, "val_loss": best_loss}, save_path)

    del model
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    return best_loss


def train_loras():
    """Train all LoRA adapters"""
    print("\n" + "=" * 60)
    print("PHASE 2: LORA TRAINING")
    print("=" * 60)

    os.makedirs(EXPERTS_DIR, exist_ok=True)
    results = {}

    for domain in EXPERTS:
        try:
            loss = train_lora(domain)
            results[domain] = loss
        except Exception as e:
            print(f"    Error: {e}")
            results[domain] = None

    print("\nLoRA Training Results:")
    for domain, loss in results.items():
        if loss:
            print(f"  {domain}: {loss:.4f}")
        else:
            print(f"  {domain}: FAILED")

    return results


class RouterDataset(Dataset):
    def __init__(self, data, vocab, max_len=128):
        self.data = data
        self.vocab = vocab
        self.max_len = max_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        tokens = re.findall(r"\b\w+\b", item["text"].lower())
        indices = [self.vocab.get(t, 1) for t in tokens[: self.max_len]]
        indices = indices + [0] * (self.max_len - len(indices))
        return {"input": torch.tensor(indices), "label": item["label"]}


class Router(nn.Module):
    def __init__(self, vocab_size, embed=128, hidden=256, num_classes=8):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, embed, padding_idx=0)
        self.conv1 = nn.Conv1d(embed, hidden, 3, padding=1)
        self.conv2 = nn.Conv1d(hidden, hidden, 3, padding=1)
        self.pool = nn.AdaptiveMaxPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden // 2, num_classes),
        )

    def forward(self, x):
        x = self.emb(x).permute(0, 2, 1)
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        return self.fc(self.pool(x).squeeze(-1))


def train_router():
    """Train routing classifier"""
    print("\n" + "=" * 60)
    print("PHASE 3: ROUTER TRAINING")
    print("=" * 60)

    label2id = {d: i for i, d in enumerate(EXPERTS)}

    train_data = []
    for domain in EXPERTS:
        path = f"{DATA_DIR}/{domain}_train.json"
        if os.path.exists(path):
            with open(path) as f:
                samples = json.load(f)[:2000]
            for s in samples:
                train_data.append({"text": s[:400], "label": label2id[domain]})

    if len(train_data) < 100:
        print("Not enough data for router")
        return False

    random.shuffle(train_data)
    n = len(train_data)
    train_set = train_data[: int(n * 0.9)]
    val_set = train_data[int(n * 0.9) :]

    vocab = {}
    idx = 2
    for item in train_set:
        for token in re.findall(r"\b\w+\b", item["text"].lower()):
            if token not in vocab and idx < 15000:
                vocab[token] = idx
                idx += 1

    train_ds = RouterDataset(train_set, vocab)
    val_ds = RouterDataset(val_set, vocab)

    train_dl = DataLoader(train_ds, batch_size=32, shuffle=True)
    val_dl = DataLoader(val_ds, batch_size=64)

    model = Router(len(vocab), num_classes=len(EXPERTS))

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    best_acc = 0

    for epoch in range(10):
        model.train()
        for batch in train_dl:
            inputs = batch["input"]
            labels = batch["label"]

            optimizer.zero_grad()
            logits = model(inputs)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

        model.eval()
        preds, labels = [], []
        with torch.no_grad():
            for batch in val_dl:
                logits = model(batch["input"])
                pred = torch.argmax(logits, dim=1)
                preds.extend(pred.numpy())
                labels.extend(batch["label"].numpy())

        acc = accuracy_score(labels, preds)
        print(f"  Epoch {epoch + 1}: {acc:.4f}")

        if acc > best_acc:
            best_acc = acc
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "vocab": vocab,
                    "config": {"vocab_size": len(vocab), "num_classes": len(EXPERTS)},
                    "experts": EXPERTS,
                },
                f"{EXPERTS_DIR}/lightweight_router.pt",
            )

    print(f"  Best accuracy: {best_acc:.4f}")
    return best_acc


def evaluate():
    """Quick evaluation"""
    print("\n" + "=" * 60)
    print("PHASE 4: EVALUATION")
    print("=" * 60)

    device = torch.device("cpu")

    ckpt = torch.load(f"{EXPERTS_DIR}/lightweight_router.pt", weights_only=False)
    router = Router(
        ckpt["config"]["vocab_size"], num_classes=ckpt["config"]["num_classes"]
    )
    router.load_state_dict(ckpt["model_state"])
    router.eval()
    vocab = ckpt["vocab"]
    experts = ckpt["experts"]

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(BASE_MODEL, torch_dtype=torch.float32)
    model = model.to(device)

    tests = [
        ("What is 15 + 27?", "math"),
        ("def hello():", "code"),
        ("Why do birds migrate?", "reasoning"),
        ("Hello, how are you?", "conversation"),
        ("What is photosynthesis?", "knowledge"),
    ]

    correct = 0
    for query, expected in tests:
        tokens = re.findall(r"\b\w+\b", query.lower())
        indices = [vocab.get(t, 1) for t in tokens[:128]]
        indices = indices + [0] * (128 - len(indices))
        inp = torch.tensor([indices])

        with torch.no_grad():
            logits = router(inp)
            pred = torch.argmax(logits, dim=1).item()
            predicted = experts[pred]

        status = "OK" if predicted == expected else "WRONG"
        if predicted == expected:
            correct += 1

        prompt = f"{query}\n\nAnswer:"
        ids = tokenizer(prompt, return_tensors="pt")["input_ids"]
        with torch.no_grad():
            out = model.generate(
                ids,
                max_new_tokens=30,
                temperature=0.7,
                do_sample=True,
                top_k=40,
                pad_token_id=tokenizer.pad_token_id,
            )
        answer = (
            tokenizer.decode(out[0], skip_special_tokens=True)
            .split("Answer:")[-1]
            .strip()[:80]
        )

        print(f"  [{predicted}] {query[:30]}... -> {answer[:50]}... [{status}]")

    print(
        f"\n  Router Accuracy: {correct}/{len(tests)} ({100 * correct / len(tests):.0f}%)"
    )
    return correct / len(tests)


def main():
    print("=" * 60)
    print("CONTINUOUS MoE TRAINING PIPELINE")
    print("=" * 60)

    iteration = 0
    while True:
        iteration += 1
        print(f"\n{'#' * 60}")
        print(f"ITERATION {iteration}")
        print(f"{'#' * 60}")

        try:
            stats = prepare_data()
            lora_results = train_loras()
            router_acc = train_router()
            eval_acc = evaluate()

            print(f"\n{'=' * 60}")
            print(f"ITERATION {iteration} COMPLETE")
            print(f"{'=' * 60}")
            print(f"  Data samples: {sum(stats.values())}")
            print(f"  Router accuracy: {router_acc:.2%}")
            print(f"  Eval accuracy: {eval_acc:.0%}")

        except Exception as e:
            print(f"\nError in iteration {iteration}: {e}")
            import traceback

            traceback.print_exc()

        print(f"\nWaiting 30 seconds before next iteration...")
        print("Press Ctrl+C to stop")
        try:
            time.sleep(30)
        except KeyboardInterrupt:
            print("\nStopping.")
            break


if __name__ == "__main__":
    main()
