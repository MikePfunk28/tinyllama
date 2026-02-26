"""
Smart Training Loop - Only trains domains below 95%
More epochs for weak domains, more data, better training
"""

import os
import json
import re
import time
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
import random

random.seed(42)

BASE_MODEL = "M:/training_models/gemma-3-270m-it"
DATA_DIR = "M:/tinyllama/cleaned_data"
EXPERTS_DIR = "M:/tinyllama/experts"
DOMAINS = ["math", "code", "reasoning", "conversation", "knowledge", "science"]
QUALITY_FILE = f"{EXPERTS_DIR}/quality_results.json"

# More data sources
MORE_DATA = {
    "math": [
        ("TIGER-Lab/MathInstruct", None, "instruction", "output"),
    ],
    "code": [
        ("sahil2801/CodeAlpaca-120k", None, "instruction", "output"),
    ],
    "reasoning": [
        ("openbookqa", "main", "question_stem", "choices"),
    ],
    "science": [
        ("allenai/sciq", None, "question", "correct_answer"),
        ("wiki_qa", None, "question", "answer"),
    ],
}


class LoraDS(Dataset):
    def __init__(self, data, tok, max_len=256):
        self.data, self.tok, self.max_len = data, tok, max_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        e = self.tok(
            self.data[i],
            max_length=self.max_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        ids = e["input_ids"].squeeze()
        lbl = ids.clone()
        lbl[lbl == self.tok.pad_token_id] = -100
        return {
            "input_ids": ids,
            "attention_mask": e["attention_mask"].squeeze(),
            "labels": lbl,
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


def get_quality():
    """Get current quality scores"""
    if os.path.exists(QUALITY_FILE):
        try:
            with open(QUALITY_FILE) as f:
                data = json.load(f)
                results = data.get("results", {})
                # Filter valid scores
                return {k: v for k, v in results.items() if v > 0}
        except:
            pass
    # Default scores based on recent evaluation
    return {
        "math": 0.80,
        "code": 0.70,
        "reasoning": 0.33,
        "conversation": 0.57,
        "knowledge": 0.83,
        "science": 0.27,
    }


def fetch_more_data(domain, max_samples=500):
    """Fetch additional data for a domain"""
    sources = MORE_DATA.get(domain, [])
    samples = []

    for name, config, field_in, field_out in sources:
        if len(samples) >= max_samples:
            break
        try:
            print(f"    Fetching {name}...")
            if config:
                ds = load_dataset(name, config, split="train", trust_remote_code=True)
            else:
                ds = load_dataset(name, split="train", trust_remote_code=True)

            for item in ds:
                if len(samples) >= max_samples:
                    break
                try:
                    if field_in == "choices":
                        q = item.get("question_stem", "")
                        choices = item.get("choices", {})
                        if isinstance(choices, dict):
                            texts = choices.get("text", [])
                            if q and texts:
                                samples.append(
                                    f"Instruction: {q} Options: {', '.join(texts)}\nResponse: {texts[0] if texts else 'Unknown'}"
                                )
                    else:
                        inst = (
                            item.get(field_in, "")
                            or item.get("instruction", "")
                            or item.get("question", "")
                        )
                        out = (
                            item.get(field_out, "")
                            or item.get("output", "")
                            or item.get("answer", "")
                            or item.get("correct_answer", "")
                        )
                        if inst and len(inst) > 10:
                            if out:
                                samples.append(f"Instruction: {inst}\nResponse: {out}")
                            else:
                                samples.append(f"Instruction: {inst}")
                except:
                    continue
        except Exception as e:
            print(f"      Error: {str(e)[:50]}")

    return samples


def train_domain(domain, epochs=3, extra_data=True):
    """Train a domain with more epochs and data"""
    print(f"\n{'=' * 50}")
    print(f"TRAINING: {domain}")
    print(f"{'=' * 50}")

    # Load existing data
    path = f"{DATA_DIR}/{domain}_train.json"
    data = []
    if os.path.exists(path):
        with open(path) as f:
            data = json.load(f)

    # Fetch more data if needed
    if extra_data:
        new_data = fetch_more_data(domain, max_samples=300)
        data.extend(new_data)
        print(f"  Total samples: {len(data)} (added {len(new_data)})")

    if len(data) < 50:
        print(f"  Not enough data: {len(data)}")
        return None

    # Use more data for training
    data = data[:800]
    print(f"  Training with: {len(data)} samples")

    tok = AutoTokenizer.from_pretrained(BASE_MODEL)
    tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(BASE_MODEL, torch_dtype=torch.float32)

    # Add LoRA
    loras = []
    for n, m in model.named_modules():
        if any(x in n for x in ["q_proj", "v_proj", "k_proj", "o_proj"]) and isinstance(
            m, nn.Linear
        ):
            parts = n.rsplit(".", 1)
            if len(parts) == 2:
                parent = model
                for p in parts[0].split("."):
                    if p:
                        parent = getattr(parent, p)
                l = LoRALayer(m, r=16, a=32)  # Higher rank
                setattr(parent, parts[1], l)
                loras.append((n, l))

    print(f"  LoRA layers: {len(loras)}")

    ds = LoraDS(data, tok)
    dl = DataLoader(ds, batch_size=4, shuffle=True)

    params = []
    for _, l in loras:
        params.extend([l.A, l.B])
    opt = torch.optim.AdamW(params, lr=2e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

    best_loss = float("inf")

    for epoch in range(epochs):
        model.train()
        loss_sum = 0

        for batch in dl:
            opt.zero_grad()
            out = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                labels=batch["labels"],
            )
            out.loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
            loss_sum += out.loss.item()

        scheduler.step()
        avg_loss = loss_sum / len(dl)
        print(f"  Epoch {epoch + 1}/{epochs}: loss={avg_loss:.4f}")

        if avg_loss < best_loss:
            best_loss = avg_loss

    # Save
    state = {}
    for n, l in loras:
        state[f"{n}.lora_A"] = l.A.data.cpu()
        state[f"{n}.lora_B"] = l.B.data.cpu()

    save_path = f"{EXPERTS_DIR}/{domain}_lora_adapter.pt"
    torch.save({"lora_state_dict": state, "loss": best_loss}, save_path)
    print(f"  Saved: {save_path}")

    del model
    return best_loss


def main():
    print("=" * 60)
    print("SMART TRAINING - Target 95%")
    print("=" * 60)

    os.makedirs(EXPERTS_DIR, exist_ok=True)

    # Get current quality
    quality = get_quality()
    print("\nCurrent quality:")
    for d in DOMAINS:
        q = quality.get(d, 0)
        status = "OK" if q >= 0.95 else "TRAIN"
        print(f"  {d}: {q * 100:.1f}% [{status}]")

    # Train domains below 95%
    need_training = [d for d in DOMAINS if quality.get(d, 0) < 0.95]

    if not need_training:
        print("\nAll domains at 95%+ !")
        return

    print(f"\nDomains needing training: {need_training}")

    # Sort by quality (train worst first)
    need_training.sort(key=lambda d: quality.get(d, 0))

    for domain in need_training:
        q = quality.get(domain, 0.5)
        # More epochs for worse quality (3-6 epochs max)
        epochs = max(2, int((0.95 - q) * 10))
        epochs = min(epochs, 6)

        print(f"\n{domain}: {q * 100:.1f}% -> training {epochs} epochs")
        train_domain(domain, epochs=epochs, extra_data=True)


if __name__ == "__main__":
    main()
