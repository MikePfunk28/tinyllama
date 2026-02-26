import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel
import json
import os
from sklearn.metrics import accuracy_score, classification_report
import numpy as np

EXPERTS = ["math", "code", "reasoning", "conversation", "knowledge"]
LABEL2ID = {label: i for i, label in enumerate(EXPERTS)}
ID2LABEL = {i: label for i, label in enumerate(EXPERTS)}

DATA_DIR = "M:/tinyllama/cleaned_data"
MODEL_PATH = "M:/training_models/gemma-3-270m-it"
SAVE_PATH = "M:/tinyllama/experts/neural_router.pt"


class RouterDataset(Dataset):
    def __init__(self, data, tokenizer, max_len=256):
        self.data = data
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        text = item["text"]
        label = LABEL2ID[item["label"]]

        encoding = self.tokenizer(
            text,
            max_length=self.max_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        return {
            "input_ids": encoding["input_ids"].squeeze(),
            "attention_mask": encoding["attention_mask"].squeeze(),
            "label": torch.tensor(label, dtype=torch.long),
        }


class NeuralRouter(nn.Module):
    def __init__(self, base_model, hidden_size=256):
        super().__init__()
        self.encoder = base_model
        self.hidden_size = self.encoder.config.hidden_size

        self.classifier = nn.Sequential(
            nn.Linear(self.hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_size // 2, len(EXPERTS)),
        )

    def forward(self, input_ids, attention_mask):
        outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
        )

        last_hidden = outputs.last_hidden_state
        mask = attention_mask.unsqueeze(-1).expand(last_hidden.size()).float()
        pooled = (last_hidden * mask).sum(1) / mask.sum(1).clamp(min=1e-9)

        logits = self.classifier(pooled)
        return logits


def load_data():
    with open(f"{DATA_DIR}/router_train.json") as f:
        train_data = json.load(f)
    with open(f"{DATA_DIR}/router_val.json") as f:
        val_data = json.load(f)
    return train_data, val_data


def train():
    print("=" * 60)
    print("TRAINING NEURAL ROUTER")
    print("=" * 60)

    device = torch.device("cpu")
    print(f"Device: {device}")

    print("\nLoading tokenizer and base model...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    tokenizer.pad_token = tokenizer.eos_token

    base_model = AutoModel.from_pretrained(MODEL_PATH)
    base_model = base_model.to(device)

    for param in base_model.parameters():
        param.requires_grad = False

    router = NeuralRouter(base_model)
    router = router.to(device)

    print(
        f"Router parameters: {sum(p.numel() for p in router.classifier.parameters()):,}"
    )

    print("\nLoading data...")
    train_data, val_data = load_data()
    print(f"Train: {len(train_data)}, Val: {len(val_data)}")

    train_dataset = RouterDataset(train_data, tokenizer)
    val_dataset = RouterDataset(val_data, tokenizer)

    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(router.classifier.parameters(), lr=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=2, factor=0.5)

    best_acc = 0
    epochs = 10

    print("\nTraining...")
    for epoch in range(epochs):
        router.train()
        total_loss = 0

        for i, batch in enumerate(train_loader):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)

            optimizer.zero_grad()
            logits = router(input_ids, attention_mask)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            if (i + 1) % 100 == 0:
                print(
                    f"  Epoch {epoch + 1}, Batch {i + 1}/{len(train_loader)}, Loss: {loss.item():.4f}"
                )

        avg_loss = total_loss / len(train_loader)

        router.eval()
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["label"].to(device)

                logits = router(input_ids, attention_mask)
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
                    "classifier_state": router.classifier.state_dict(),
                    "config": {
                        "hidden_size": router.hidden_size,
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
    router.classifier.load_state_dict(checkpoint["classifier_state"])
    router.eval()

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)

            logits = router(input_ids, attention_mask)
            preds = torch.argmax(logits, dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    print(classification_report(all_labels, all_preds, target_names=EXPERTS))
    print(f"Best Validation Accuracy: {best_acc:.4f}")
    print(f"\nRouter saved to: {SAVE_PATH}")


if __name__ == "__main__":
    train()
