"""
Fast training - Just train weak domains quickly
"""

import os
import json
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer
import random

random.seed(42)

BASE = "M:/training_models/gemma-3-270m-it"
DATA = "M:/tinyllama/cleaned_data"
EXP = "M:/tinyllama/experts"
QUAL = f"{EXP}/quality_results.json"
DOMAINS = ["math", "code", "reasoning", "conversation", "knowledge", "science"]


class DS(Dataset):
    def __init__(self, data, tok, ml=256):
        self.data, self.tok, self.ml = data, tok, ml

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        e = self.tok(
            self.data[i],
            max_length=self.ml,
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


class LoRA(nn.Module):
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
    if os.path.exists(QUAL):
        with open(QUAL) as f:
            d = json.load(f)
            return d.get("results", {})
    return {}


def train(domain, epochs=1):
    print(f"\n[{domain}]")

    path = f"{DATA}/{domain}_train.json"
    if not os.path.exists(path):
        print("  No data")
        return

    with open(path) as f:
        data = json.load(f)[:150]  # Reduced samples for faster training

    if len(data) < 20:
        print(f"  Not enough: {len(data)}")
        return

    print(f"  Samples: {len(data)}")

    tok = AutoTokenizer.from_pretrained(BASE)
    tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(BASE, torch_dtype=torch.float32)

    loras = []
    for n, m in model.named_modules():
        if any(x in n for x in ["q_proj", "v_proj"]) and isinstance(m, nn.Linear):
            parts = n.rsplit(".", 1)
            if len(parts) == 2:
                parent = model
                for p in parts[0].split("."):
                    if p:
                        parent = getattr(parent, p)
                l = LoRA(m)
                setattr(parent, parts[1], l)
                loras.append((n, l))

    ds = DS(data, tok)
    dl = DataLoader(ds, batch_size=4, shuffle=True)

    params = []
    for _, l in loras:
        params.extend([l.A, l.B])
    opt = torch.optim.AdamW(params, lr=1e-4)

    model.train()
    for ep in range(epochs):
        loss = 0
        for b in dl:
            opt.zero_grad()
            out = model(
                input_ids=b["input_ids"],
                attention_mask=b["attention_mask"],
                labels=b["labels"],
            )
            out.loss.backward()
            opt.step()
            loss += out.loss.item()
        print(f"  Ep {ep + 1}: {loss / len(dl):.3f}")

    state = {}
    for n, l in loras:
        state[f"{n}.lora_A"] = l.A.data.cpu()
        state[f"{n}.lora_B"] = l.B.data.cpu()

    torch.save({"lora_state_dict": state}, f"{EXP}/{domain}_lora_adapter.pt")
    print(f"  Saved")

    del model


def main():
    print("FAST TRAINING")

    q = get_quality()
    print(f"Current quality:")
    for d, v in sorted(q.items(), key=lambda x: x[1]):
        print(f"  {d}: {v * 100:.0f}%")

    # Train only 1 weakest domain
    weak = [d for d in DOMAINS if q.get(d, 0) < 0.8]
    print(f"\nWeak domains: {weak}")

    if weak:
        weakest = min(weak, key=lambda x: q.get(x, 0))
        print(f"Training: {weakest}")
        train(weakest, epochs=1)

    print("\nDone")


if __name__ == "__main__":
    main()
