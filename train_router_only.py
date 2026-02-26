"""Train router only - fix for vocab issue"""

import os
import json
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import re

DATA_DIR = "M:/tinyllama/cleaned_data"
EXPERTS_DIR = "M:/tinyllama/experts"
DOMAINS = ["math", "code", "reasoning", "conversation", "knowledge", "science"]


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
        x = self.emb(x).permute(0, 2, 1)
        x = torch.relu(self.conv(x))
        return self.fc(self.pool(x).squeeze(-1))


class RDS(Dataset):
    def __init__(self, data, vocab, max_len=100):
        self.data, self.vocab, self.max_len = data, vocab, max_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        tokens = re.findall(r"\b\w+\b", self.data[i]["text"].lower())
        ids = [self.vocab.get(t, 1) for t in tokens[: self.max_len]]
        ids = ids + [0] * (self.max_len - len(ids))
        return {"x": torch.tensor(ids), "y": self.data[i]["label"]}


def main():
    print("Training Router...")

    try:
        from sklearn.metrics import accuracy_score
    except ImportError:
        print("sklearn not available")
        return

    label2id = {d: i for i, d in enumerate(DOMAINS)}

    train_data = []
    for domain in DOMAINS:
        path = f"{DATA_DIR}/{domain}_train.json"
        if os.path.exists(path):
            with open(path) as f:
                samples = json.load(f)[:2000]
            for s in samples:
                if len(s) > 30:
                    train_data.append({"text": s[:300], "label": label2id[domain]})

    print(f"Samples: {len(train_data)}")

    # Build vocab with buffer
    vocab = {"<PAD>": 0, "<UNK>": 1}
    idx = 2
    for item in train_data:
        for token in re.findall(r"\b\w+\b", item["text"].lower()):
            if token not in vocab and idx < 15000:
                vocab[token] = idx
                idx += 1

    print(f"Vocab size: {len(vocab)}")

    import random

    random.shuffle(train_data)
    split = int(0.9 * len(train_data))

    train_ds = RDS(train_data[:split], vocab)
    val_ds = RDS(train_data[split:], vocab)
    train_dl = DataLoader(train_ds, batch_size=32, shuffle=True)
    val_dl = DataLoader(val_ds, batch_size=64)

    model = Router(len(vocab), len(DOMAINS))
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    best_acc = 0
    for epoch in range(10):
        model.train()
        for batch in train_dl:
            opt.zero_grad()
            loss = criterion(model(batch["x"]), batch["y"])
            loss.backward()
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

    print(f"Best: {best_acc:.4f}")
    print(f"Saved: {EXPERTS_DIR}/lightweight_router.pt")


if __name__ == "__main__":
    main()
