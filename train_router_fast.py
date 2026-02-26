import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
import json
import os
from sklearn.metrics import accuracy_score, classification_report
import numpy as np
from collections import Counter
import re

EXPERTS = ["math", "code", "reasoning", "conversation", "knowledge"]
LABEL2ID = {label: i for i, label in enumerate(EXPERTS)}
ID2LABEL = {i: label for i, label in enumerate(EXPERTS)}

DATA_DIR = "M:/tinyllama/cleaned_data"
SAVE_PATH = "M:/tinyllama/experts/lightweight_router.pt"


class TextVectorizer:
    def __init__(self, max_features=10000, max_len=128):
        self.max_features = max_features
        self.max_len = max_len
        self.vocab = {}
        self.idf = {}

    def tokenize(self, text):
        text = text.lower()
        tokens = re.findall(r"\b\w+\b", text)
        return tokens

    def fit(self, texts):
        doc_freq = Counter()
        all_tokens = set()

        for text in texts:
            tokens = set(self.tokenize(text))
            for t in tokens:
                doc_freq[t] += 1
            all_tokens.update(tokens)

        top_tokens = sorted(all_tokens, key=lambda t: doc_freq[t], reverse=True)[
            : self.max_features - 2
        ]
        self.vocab = {"<PAD>": 0, "<UNK>": 1}
        for i, t in enumerate(top_tokens):
            self.vocab[t] = i + 2

        n_docs = len(texts)
        self.idf = {}
        for t, freq in doc_freq.items():
            if t in self.vocab:
                self.idf[self.vocab[t]] = np.log(n_docs / (1 + freq))

        print(f"Vocab size: {len(self.vocab)}")

    def transform(self, texts):
        vectors = []
        for text in texts:
            tokens = self.tokenize(text)
            indices = [self.vocab.get(t, 1) for t in tokens[: self.max_len]]
            indices = indices + [0] * (self.max_len - len(indices))
            vectors.append(indices)
        return np.array(vectors)


class FastDataset(Dataset):
    def __init__(self, texts, labels, vectorizer):
        self.texts = vectorizer.transform(texts)
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {
            "text": torch.tensor(self.texts[idx], dtype=torch.long),
            "label": torch.tensor(self.labels[idx], dtype=torch.long),
        }


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


def load_data():
    with open(f"{DATA_DIR}/router_train.json") as f:
        train_data = json.load(f)
    with open(f"{DATA_DIR}/router_val.json") as f:
        val_data = json.load(f)

    train_texts = [d["text"] for d in train_data]
    train_labels = [LABEL2ID[d["label"]] for d in train_data]
    val_texts = [d["text"] for d in val_data]
    val_labels = [LABEL2ID[d["label"]] for d in val_data]

    return train_texts, train_labels, val_texts, val_labels


def train():
    print("=" * 60)
    print("TRAINING LIGHTWEIGHT ROUTER")
    print("=" * 60)

    device = torch.device("cpu")
    print(f"Device: {device}")

    print("\nLoading data...")
    train_texts, train_labels, val_texts, val_labels = load_data()
    print(f"Train: {len(train_texts)}, Val: {len(val_texts)}")

    print("\nBuilding vocabulary...")
    vectorizer = TextVectorizer(max_features=10000, max_len=128)
    vectorizer.fit(train_texts)

    train_dataset = FastDataset(train_texts, train_labels, vectorizer)
    val_dataset = FastDataset(val_texts, val_labels, vectorizer)

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=128)

    model = FastRouter(
        vocab_size=len(vectorizer.vocab),
        embed_dim=128,
        hidden_dim=256,
        num_classes=len(EXPERTS),
    )
    model = model.to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {total_params:,}")

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=2, factor=0.5)

    best_acc = 0
    epochs = 15

    print("\nTraining...")
    for epoch in range(epochs):
        model.train()
        total_loss = 0

        for i, batch in enumerate(train_loader):
            texts = batch["text"].to(device)
            labels = batch["label"].to(device)

            optimizer.zero_grad()
            logits = model(texts)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            if (i + 1) % 50 == 0:
                print(
                    f"  Epoch {epoch + 1}, Batch {i + 1}/{len(train_loader)}, Loss: {loss.item():.4f}"
                )

        avg_loss = total_loss / len(train_loader)

        model.eval()
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for batch in val_loader:
                texts = batch["text"].to(device)
                labels = batch["label"].to(device)

                logits = model(texts)
                preds = torch.argmax(logits, dim=1)

                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        acc = accuracy_score(all_labels, all_preds)
        scheduler.step(1 - acc)

        print(f"Epoch {epoch + 1}/{epochs} - Loss: {avg_loss:.4f}, Val Acc: {acc:.4f}")

        if acc > best_acc:
            best_acc = acc
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "vocab": vectorizer.vocab,
                    "config": {
                        "vocab_size": len(vectorizer.vocab),
                        "embed_dim": 128,
                        "hidden_dim": 256,
                        "num_classes": len(EXPERTS),
                    },
                },
                SAVE_PATH,
            )
            print(f"  -> Saved best model (acc: {acc:.4f})")

    print("\n" + "=" * 60)
    print("FINAL EVALUATION")
    print("=" * 60)

    checkpoint = torch.load(SAVE_PATH, weights_only=False)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in val_loader:
            texts = batch["text"].to(device)
            labels = batch["label"].to(device)

            logits = model(texts)
            preds = torch.argmax(logits, dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    print(classification_report(all_labels, all_preds, target_names=EXPERTS))
    print(f"Best Validation Accuracy: {best_acc:.4f}")
    print(f"\nRouter saved to: {SAVE_PATH}")


if __name__ == "__main__":
    train()
