"""
Resumable LoRA Training with Checkpoints
- Saves progress after each domain
- Can resume from interruption
- Checkpoints every N batches
"""

import os
import json
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer
import glob
import re

BASE_MODEL = "M:/training_models/gemma-3-270m-it"
DATA_DIR = "M:/tinyllama/cleaned_data"
EXPERTS_DIR = "M:/tinyllama/experts"
CHECKPOINT_FILE = "M:/tinyllama/experts/training_checkpoint.json"
DOMAINS = ["math", "code", "reasoning", "conversation", "knowledge", "science"]


class LoraDS(Dataset):
    def __init__(self, data, tok, max_len=256):
        self.data = data
        self.tok = tok
        self.max_len = max_len

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
        self.orig = orig
        self.r, self.a = r, a
        self.A = nn.Parameter(torch.zeros(r, orig.in_features))
        self.B = nn.Parameter(torch.zeros(orig.out_features, r))
        nn.init.kaiming_uniform_(self.A, a=5**0.5)
        for p in orig.parameters():
            p.requires_grad = False

    def forward(self, x):
        return self.orig(x) + (self.a / self.r) * (x @ self.A.T @ self.B.T)


def load_checkpoint():
    if os.path.exists(CHECKPOINT_FILE):
        with open(CHECKPOINT_FILE) as f:
            return json.load(f)
    return {"completed": [], "current": None, "epoch": 0, "batch": 0}


def save_checkpoint(data):
    with open(CHECKPOINT_FILE, "w") as f:
        json.dump(data, f, indent=2)


def save_lora_ckpt(domain, loras, epoch, batch, loss):
    state = {f"{n}.lora_A": l.A.data.cpu() for n, l in loras}
    for n, l in loras:
        state[f"{n}.lora_B"] = l.B.data.cpu()
    torch.save(
        {"lora_state_dict": state, "epoch": epoch, "batch": batch, "loss": loss},
        f"{EXPERTS_DIR}/{domain}_checkpoint.pt",
    )


def save_lora_final(domain, loras, loss):
    state = {f"{n}.lora_A": l.A.data.cpu() for n, l in loras}
    for n, l in loras:
        state[f"{n}.lora_B"] = l.B.data.cpu()
    path = f"{EXPERTS_DIR}/{domain}_lora_adapter.pt"
    torch.save({"lora_state_dict": state, "final_loss": loss}, path)
    return path


def load_lora_ckpt(domain, model):
    ckpt_path = f"{EXPERTS_DIR}/{domain}_checkpoint.pt"

    loras = []
    for n, m in model.named_modules():
        if any(x in n for x in ["q_proj", "v_proj"]) and isinstance(m, nn.Linear):
            parts = n.rsplit(".", 1)
            if len(parts) == 2:
                parent = model
                for p in parts[0].split("."):
                    if p:
                        parent = getattr(parent, p)
                l = LoRALayer(m)
                setattr(parent, parts[1], l)
                loras.append((n, l))

    if os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        state = ckpt.get("lora_state_dict", {})
        for n, l in loras:
            if f"{n}.lora_A" in state:
                l.A.data = state[f"{n}.lora_A"]
                l.B.data = state[f"{n}.lora_B"]
        print(f"  Resumed: epoch {ckpt.get('epoch', 0)}, batch {ckpt.get('batch', 0)}")
        return loras, ckpt.get("epoch", 0), ckpt.get("batch", 0), ckpt.get("loss", 999)

    return loras, 0, 0, 999


def train_domain(domain, resume=True):
    print(f"\n{'=' * 50}")
    print(f"TRAINING: {domain}")
    print(f"{'=' * 50}")

    ckpt = load_checkpoint()
    if domain in ckpt.get("completed", []) and resume:
        print(f"  Already completed, skipping")
        return True

    path = f"{DATA_DIR}/{domain}_train.json"
    if not os.path.exists(path):
        print(f"  No training data")
        return False

    with open(path) as f:
        data = json.load(f)[:1000]

    if len(data) < 30:
        print(f"  Not enough data: {len(data)}")
        return False

    print(f"  Samples: {len(data)}")

    tok = AutoTokenizer.from_pretrained(BASE_MODEL)
    tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(BASE_MODEL, torch_dtype=torch.float32)

    loras, start_epoch, start_batch, best_loss = (
        load_lora_ckpt(domain, model) if resume else (None, 0, 0, 999)
    )
    if loras is None:
        loras = []
        for n, m in model.named_modules():
            if any(x in n for x in ["q_proj", "v_proj"]) and isinstance(m, nn.Linear):
                parts = n.rsplit(".", 1)
                if len(parts) == 2:
                    parent = model
                    for p in parts[0].split("."):
                        if p:
                            parent = getattr(parent, p)
                    loras.append((n, LoRALayer(m)))
                    setattr(parent, parts[1], loras[-1][1])
        best_loss = 999

    print(f"  LoRA layers: {len(loras)}")

    ds = LoraDS(data, tok)
    dl = DataLoader(ds, batch_size=2, shuffle=True)

    params = []
    for _, l in loras:
        params.extend([l.A, l.B])
    opt = torch.optim.AdamW(params, lr=1e-4)

    epochs = 3
    ckpt_every = 100

    model.train()
    for epoch in range(start_epoch, epochs):
        loss_sum = 0
        for bi, batch in enumerate(dl):
            if epoch == start_epoch and bi < start_batch:
                continue

            opt.zero_grad()
            out = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                labels=batch["labels"],
            )
            out.loss.backward()
            opt.step()
            loss_sum += out.loss.item()

            if (bi + 1) % ckpt_every == 0:
                avg = loss_sum / (bi + 1)
                save_lora_ckpt(domain, loras, epoch, bi + 1, avg)
                global_ckpt = load_checkpoint()
                global_ckpt.update({"current": domain, "epoch": epoch, "batch": bi + 1})
                save_checkpoint(global_ckpt)

        avg_loss = loss_sum / len(dl)
        print(f"  Epoch {epoch + 1}/{epochs}: loss={avg_loss:.4f}")
        save_lora_ckpt(domain, loras, epoch + 1, 0, avg_loss)

        if avg_loss < best_loss:
            best_loss = avg_loss

    final_path = save_lora_final(domain, loras, best_loss)
    print(f"  Saved: {final_path} (loss={best_loss:.4f})")

    global_ckpt = load_checkpoint()
    if domain not in global_ckpt.get("completed", []):
        global_ckpt["completed"].append(domain)
    global_ckpt["current"] = None
    save_checkpoint(global_ckpt)

    ckpt_path = f"{EXPERTS_DIR}/{domain}_checkpoint.pt"
    if os.path.exists(ckpt_path):
        os.remove(ckpt_path)

    del model
    return True


def train_router():
    print(f"\n{'=' * 50}")
    print("TRAINING ROUTER")
    print(f"{'=' * 50}")

    try:
        from sklearn.metrics import accuracy_score
    except:
        print("  sklearn not available, skipping")
        return False

    label2id = {d: i for i, d in enumerate(DOMAINS)}

    train_data = []
    for domain in DOMAINS:
        path = f"{DATA_DIR}/{domain}_train.json"
        if os.path.exists(path):
            with open(path) as f:
                samples = json.load(f)[:1500]
            for s in samples:
                train_data.append({"text": s[:300], "label": label2id[domain]})

    if len(train_data) < 50:
        print("  Not enough data")
        return False

    print(f"  Samples: {len(train_data)}")

    vocab = {}
    idx = 2
    for item in train_data:
        for token in re.findall(r"\b\w+\b", item["text"].lower()):
            if token not in vocab and idx < 12000:
                vocab[token] = idx
                idx += 1

    class Router(nn.Module):
        def __init__(self, vs, nc):
            super().__init__()
            self.emb = nn.Embedding(vs, 128, padding_idx=0)
            self.conv = nn.Conv1d(128, 256, 3, padding=1)
            self.pool = nn.AdaptiveMaxPool1d(1)
            self.fc = nn.Sequential(
                nn.Linear(256, 128), nn.ReLU(), nn.Dropout(0.3), nn.Linear(128, nc)
            )

        def forward(self, x):
            return self.fc(
                self.pool(torch.relu(self.conv(self.emb(x).permute(0, 2, 1)))).squeeze(
                    -1
                )
            )

    model = Router(len(vocab), len(DOMAINS))

    class RDS(Dataset):
        def __init__(self, data, vocab):
            self.data, self.vocab = data, vocab

        def __len__(self):
            return len(self.data)

        def __getitem__(self, i):
            tokens = re.findall(r"\b\w+\b", self.data[i]["text"].lower())
            ids = [self.vocab.get(t, 1) for t in tokens[:100]] + [0] * (
                100 - len(tokens[:100])
            )
            return {"x": torch.tensor(ids), "y": self.data[i]["label"]}

    import random

    random.shuffle(train_data)
    split = int(0.9 * len(train_data))
    train_dl = DataLoader(RDS(train_data[:split], vocab), batch_size=32, shuffle=True)
    val_dl = DataLoader(RDS(train_data[split:], vocab), batch_size=64)

    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    best_acc = 0
    for epoch in range(8):
        model.train()
        for batch in train_dl:
            opt.zero_grad()
            criterion(model(batch["x"]), batch["y"]).backward()
            opt.step()

        model.eval()
        preds, labels = [], []
        with torch.no_grad():
            for batch in val_dl:
                out = model(batch["x"])
                preds.extend(torch.argmax(out, dim=1).numpy())
                labels.extend(batch["y"].numpy())

        acc = accuracy_score(labels, preds)
        print(f"  Epoch {epoch + 1}: acc={acc:.4f}")

        if acc > best_acc:
            best_acc = acc
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "vocab": vocab,
                    "config": {"vocab_size": len(vocab), "num_classes": len(DOMAINS)},
                    "experts": DOMAINS,
                },
                f"{EXPERTS_DIR}/lightweight_router.pt",
            )

    print(f"  Best: {best_acc:.4f}")
    return True


def main():
    print("=" * 60)
    print("RESUMABLE LORA TRAINING")
    print("=" * 60)

    os.makedirs(EXPERTS_DIR, exist_ok=True)

    ckpt = load_checkpoint()
    print(f"\nProgress: {ckpt}")

    for domain in DOMAINS:
        train_domain(domain)

    print("\n" + "=" * 60)
    print("LORA TRAINING COMPLETE")
    print("=" * 60)

    print("\nAdapters:")
    for f in sorted(glob.glob(f"{EXPERTS_DIR}/*_lora_adapter.pt")):
        print(f"  {os.path.basename(f)}: {os.path.getsize(f) // 1024} KB")


if __name__ == "__main__":
    main()
