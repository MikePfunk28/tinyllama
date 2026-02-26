import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM
import json
import os
import sys
from tqdm import tqdm

try:
    import torch_directml

    USE_DIRECTML = False  # DirectML has memory issues, use CPU
    device = torch.device("cpu")
    DEVICE_NAME = "CPU"
except:
    device = torch.device("cpu")
    DEVICE_NAME = "CPU"

EXPERTS = ["math", "code", "reasoning", "conversation", "knowledge"]
DATA_DIR = "M:/tinyllama/cleaned_data"
BASE_MODEL = "M:/training_models/gemma-3-270m-it"
SAVE_DIR = "M:/tinyllama/experts"


class LoRALayer(nn.Module):
    def __init__(self, original_layer, rank=8, alpha=16):
        super().__init__()
        self.original = original_layer
        self.rank = rank
        self.alpha = alpha

        in_features = original_layer.in_features
        out_features = original_layer.out_features

        self.lora_A = nn.Parameter(torch.zeros(rank, in_features))
        self.lora_B = nn.Parameter(torch.zeros(out_features, rank))

        nn.init.kaiming_uniform_(self.lora_A, a=5**0.5)
        nn.init.zeros_(self.lora_B)

        for param in self.original.parameters():
            param.requires_grad = False

    def forward(self, x):
        original_out = self.original(x)
        lora_out = (x @ self.lora_A.T) @ self.lora_B.T
        return original_out + (self.alpha / self.rank) * lora_out


class ExpertDataset(Dataset):
    def __init__(self, data, tokenizer, max_len=128):
        self.data = data
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        text = self.data[idx]
        encoding = self.tokenizer(
            text,
            max_length=self.max_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        input_ids = encoding["input_ids"].squeeze()
        attention_mask = encoding["attention_mask"].squeeze()
        labels = input_ids.clone()
        labels[labels == self.tokenizer.pad_token_id] = -100
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }


def add_lora_to_model(
    model, rank=8, alpha=16, target_modules=["q_proj", "v_proj", "k_proj", "o_proj"]
):
    lora_layers = []

    for name, module in model.named_modules():
        if any(target in name for target in target_modules) and isinstance(
            module, nn.Linear
        ):
            parent_name = ".".join(name.split(".")[:-1])
            child_name = name.split(".")[-1]

            parent = model
            for part in parent_name.split("."):
                if part:
                    parent = getattr(parent, part)

            lora_layer = LoRALayer(module, rank=rank, alpha=alpha)
            setattr(parent, child_name, lora_layer)
            lora_layers.append((name, lora_layer))

    return lora_layers


def get_lora_params(lora_layers):
    params = []
    for name, layer in lora_layers:
        params.extend([layer.lora_A, layer.lora_B])
    return params


def save_lora_weights(lora_layers, path):
    state_dict = {}
    for name, layer in lora_layers:
        state_dict[f"{name}.lora_A"] = layer.lora_A.data.cpu()
        state_dict[f"{name}.lora_B"] = layer.lora_B.data.cpu()
    torch.save(state_dict, path)
    print(f"  Saved LoRA weights to {path}")


def train_expert(expert_name, epochs=2, batch_size=2):
    print(f"\n{'=' * 60}")
    print(f"TRAINING EXPERT: {expert_name}")
    print(f"{'=' * 60}")

    print(f"Loading base model...")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(BASE_MODEL, torch_dtype=torch.float32)
    model = model.to(device)

    lora_layers = add_lora_to_model(model, rank=8, alpha=16)
    print(f"Added {len(lora_layers)} LoRA layers")

    lora_params = get_lora_params(lora_layers)
    trainable = sum(p.numel() for p in lora_params)
    total = sum(p.numel() for p in model.parameters())
    print(
        f"Trainable: {trainable:,} / Total: {total:,} ({100 * trainable / total:.2f}%)"
    )

    print(f"Loading {expert_name} data...")
    train_path = f"{DATA_DIR}/{expert_name}_train.json"
    val_path = f"{DATA_DIR}/{expert_name}_val.json"

    with open(train_path) as f:
        train_data = json.load(f)[:500]
    with open(val_path) as f:
        val_data = json.load(f)[:100]

    print(f"Train: {len(train_data)}, Val: {len(val_data)}")

    train_dataset = ExpertDataset(train_data, tokenizer)
    val_dataset = ExpertDataset(val_data, tokenizer)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)

    optimizer = optim.AdamW(lora_params, lr=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_loss = float("inf")

    for epoch in range(epochs):
        model.train()
        total_loss = 0

        progress = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs}")
        for batch in progress:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            optimizer.zero_grad()
            outputs = model(
                input_ids=input_ids, attention_mask=attention_mask, labels=labels
            )
            loss = outputs.loss
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            progress.set_postfix({"loss": f"{loss.item():.4f}"})

        avg_train_loss = total_loss / len(train_loader)
        scheduler.step()

        model.eval()
        val_loss = 0

        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["labels"].to(device)

                outputs = model(
                    input_ids=input_ids, attention_mask=attention_mask, labels=labels
                )
                val_loss += outputs.loss.item()

        avg_val_loss = val_loss / len(val_loader)
        print(
            f"Epoch {epoch + 1}: Train Loss={avg_train_loss:.4f}, Val Loss={avg_val_loss:.4f}"
        )

        if avg_val_loss < best_loss:
            best_loss = avg_val_loss
            save_path = f"{SAVE_DIR}/{expert_name}_lora_adapter_v2.pt"
            save_lora_weights(lora_layers, save_path)

    del model
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    return best_loss


def main():
    print("=" * 60)
    print("LoRA ADAPTER TRAINING ON CLEANED DATA")
    print(f"Device: {DEVICE_NAME}")
    print("=" * 60)

    results = {}

    for expert in EXPERTS:
        try:
            loss = train_expert(expert, epochs=2, batch_size=4)
            results[expert] = loss
        except Exception as e:
            print(f"Error training {expert}: {e}")
            results[expert] = None

    print("\n" + "=" * 60)
    print("TRAINING SUMMARY")
    print("=" * 60)
    for expert, loss in results.items():
        if loss:
            print(f"  {expert}: Val Loss = {loss:.4f}")
        else:
            print(f"  {expert}: FAILED")

    print("\nDone!")


if __name__ == "__main__":
    main()
