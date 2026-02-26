import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, random_split
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from typing import Optional, Dict, Any, List
from tqdm import tqdm
import json
import re

try:
    import torch_directml

    HAS_DIRECTML = True
except ImportError:
    HAS_DIRECTML = False

try:
    from peft import LoraConfig, get_peft_model, TaskType

    HAS_PEFT = True
except ImportError:
    HAS_PEFT = False

try:
    import bitsandbytes as bnb

    HAS_BNB = True
except ImportError:
    HAS_BNB = False


def get_device():
    if HAS_DIRECTML:
        try:
            device = torch_directml.device()
            print(f"Using DirectML device: {device}")
            return device
        except Exception as e:
            print(f"DirectML error: {e}")
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"Using CUDA: {torch.cuda.get_device_name(0)}")
        return device
    device = torch.device("cpu")
    print("Using CPU")
    return device


def normalize_text(text: str) -> str:
    text = re.sub(r"\s+", " ", text)
    text = text.strip()
    text = re.sub(r"[\x00-\x1f\x7f-\x9f]", "", text)
    return text


class TextDataset(Dataset):
    def __init__(
        self, data_path: str, tokenizer, max_length: int = 512, normalize: bool = True
    ):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.normalize = normalize
        self.examples = []
        self.attention_masks = []

        print(f"Loading data from {data_path}...")

        if data_path.endswith(".jsonl"):
            with open(data_path, "r", encoding="utf-8") as f:
                for line in tqdm(f, desc="Processing"):
                    item = json.loads(line)
                    text = item.get("text", "")
                    if not text:
                        continue

                    if self.normalize:
                        text = normalize_text(text)

                    encoding = tokenizer(
                        text,
                        max_length=max_length,
                        truncation=True,
                        padding="max_length",
                        return_tensors="pt",
                        add_special_tokens=True,
                    )

                    input_ids = encoding["input_ids"].squeeze(0)
                    attention_mask = encoding["attention_mask"].squeeze(0)

                    if attention_mask.sum() > 10:
                        self.examples.append(input_ids)
                        self.attention_masks.append(attention_mask)

        print(f"Loaded {len(self.examples)} examples")

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return {
            "input_ids": self.examples[idx],
            "attention_mask": self.attention_masks[idx],
        }


class Trainer:
    def __init__(
        self,
        model: nn.Module,
        train_dataset: Dataset,
        eval_dataset: Optional[Dataset] = None,
        tokenizer=None,
        config: Optional[Dict[str, Any]] = None,
    ):
        self.model = model
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.tokenizer = tokenizer
        self.config = config or {}

        self.device = self.config.get("device")
        if self.device is None or self.device == "auto":
            self.device = get_device()
        elif isinstance(self.device, str):
            self.device = torch.device(self.device)

        self.batch_size = self.config.get("batch_size", 4)
        self.gradient_accumulation_steps = self.config.get(
            "gradient_accumulation_steps", 4
        )
        self.learning_rate = self.config.get("learning_rate", 3e-4)
        self.weight_decay = self.config.get("weight_decay", 0.01)
        self.warmup_steps = self.config.get("warmup_steps", 100)
        self.max_steps = self.config.get("max_steps", 10000)
        self.save_steps = self.config.get("save_steps", 1000)
        self.eval_steps = self.config.get("eval_steps", 500)
        self.output_dir = self.config.get("output_dir", "./checkpoints")
        self.use_lora = self.config.get("use_lora", False)
        self.use_8bit = self.config.get("use_8bit", False)
        self.use_gradient_checkpointing = self.config.get(
            "use_gradient_checkpointing", False
        )
        self.pad_token_id = tokenizer.pad_token_id if tokenizer else 0

        os.makedirs(self.output_dir, exist_ok=True)

        self._setup_model()
        self._setup_optimizer()
        self._setup_dataloader()

    def _setup_model(self):
        if self.use_lora and HAS_PEFT:
            lora_config = LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                r=self.config.get("lora_r", 8),
                lora_alpha=self.config.get("lora_alpha", 16),
                lora_dropout=self.config.get("lora_dropout", 0.05),
                target_modules=[
                    "q_proj",
                    "k_proj",
                    "v_proj",
                    "o_proj",
                    "key",
                    "value",
                    "output",
                    "receptance",
                ],
            )
            self.model = get_peft_model(self.model, lora_config)
            print("LoRA enabled")

        if self.use_gradient_checkpointing and hasattr(
            self.model, "gradient_checkpointing_enable"
        ):
            self.model.gradient_checkpointing_enable()
            print("Gradient checkpointing enabled")

        self.model = self.model.to(self.device)
        print(f"Model parameters: {self.model.get_num_params():,}")

    def _setup_optimizer(self):
        no_decay = ["bias", "LayerNorm.weight", "layer_norm.weight", "ln_", "norm"]
        optimizer_grouped_parameters = [
            {
                "params": [
                    p
                    for n, p in self.model.named_parameters()
                    if not any(nd in n for nd in no_decay)
                ],
                "weight_decay": self.weight_decay,
            },
            {
                "params": [
                    p
                    for n, p in self.model.named_parameters()
                    if any(nd in n for nd in no_decay)
                ],
                "weight_decay": 0.0,
            },
        ]

        if self.use_8bit and HAS_BNB:
            self.optimizer = bnb.optim.AdamW8bit(
                optimizer_grouped_parameters,
                lr=self.learning_rate,
            )
            print("8-bit AdamW enabled")
        else:
            self.optimizer = AdamW(
                optimizer_grouped_parameters,
                lr=self.learning_rate,
            )

        warmup_scheduler = LinearLR(
            self.optimizer,
            start_factor=0.1,
            end_factor=1.0,
            total_iters=self.warmup_steps,
        )
        cosine_scheduler = CosineAnnealingLR(
            self.optimizer,
            T_max=self.max_steps - self.warmup_steps,
            eta_min=self.learning_rate * 0.01,
        )
        self.scheduler = SequentialLR(
            self.optimizer,
            schedulers=[warmup_scheduler, cosine_scheduler],
            milestones=[self.warmup_steps],
        )

    def _setup_dataloader(self):
        def collate_fn(batch):
            input_ids = torch.stack([item["input_ids"] for item in batch])
            attention_mask = torch.stack([item["attention_mask"] for item in batch])
            return {"input_ids": input_ids, "attention_mask": attention_mask}

        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            collate_fn=collate_fn,
            num_workers=0,
            pin_memory=True,
        )

        if self.eval_dataset:
            self.eval_loader = DataLoader(
                self.eval_dataset,
                batch_size=self.batch_size,
                collate_fn=collate_fn,
                num_workers=0,
                pin_memory=True,
            )

    def compute_loss(self, batch):
        input_ids = batch["input_ids"].to(self.device)
        attention_mask = batch["attention_mask"].to(self.device)

        labels = input_ids.clone()
        labels[attention_mask == 0] = -100
        labels[:, :-1] = labels[:, 1:]
        labels[:, -1] = -100

        inputs = input_ids[:, :-1].contiguous()
        mask = attention_mask[:, :-1].contiguous()
        targets = labels[:, :-1].contiguous()

        if "RWKV" in self.model.__class__.__name__:
            logits, _ = self.model(inputs)
        else:
            logits = self.model(inputs, mask)

        loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
        loss = loss_fct(logits.view(-1, logits.size(-1)), targets.view(-1))
        return loss

    def train(self):
        self.model.train()
        global_step = 0
        total_loss = 0.0
        best_loss = float("inf")

        epoch = 0
        while global_step < self.max_steps:
            epoch += 1
            progress_bar = tqdm(self.train_loader, desc=f"Epoch {epoch}")

            for batch in progress_bar:
                if global_step >= self.max_steps:
                    break

                loss = self.compute_loss(batch)
                loss = loss / self.gradient_accumulation_steps

                if torch.isnan(loss):
                    print("NaN loss detected, skipping batch")
                    self.optimizer.zero_grad()
                    continue

                loss.backward()

                total_loss += loss.item()

                if (global_step + 1) % self.gradient_accumulation_steps == 0:
                    grad_norm = torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), 1.0
                    )
                    self.optimizer.step()
                    self.scheduler.step()
                    self.optimizer.zero_grad()

                    avg_loss = total_loss / self.gradient_accumulation_steps
                    progress_bar.set_postfix(
                        loss=f"{avg_loss:.4f}",
                        lr=f"{self.scheduler.get_last_lr()[0]:.2e}",
                        grad_norm=f"{grad_norm:.2f}",
                    )
                    total_loss = 0.0

                global_step += 1

                if global_step % self.save_steps == 0:
                    self.save_checkpoint(global_step)

                if global_step % self.eval_steps == 0 and self.eval_dataset:
                    eval_loss = self.evaluate(global_step)
                    if eval_loss < best_loss:
                        best_loss = eval_loss
                        self.save_checkpoint(global_step, best=True)
                    self.model.train()

        self.save_checkpoint(global_step, final=True)
        print("Training completed!")

    @torch.no_grad()
    def evaluate(self, step):
        self.model.eval()
        total_loss = 0.0
        num_batches = 0

        for batch in tqdm(self.eval_loader, desc="Evaluating"):
            loss = self.compute_loss(batch)
            if not torch.isnan(loss):
                total_loss += loss.item()
                num_batches += 1

            if num_batches >= 100:
                break

        avg_loss = total_loss / max(num_batches, 1)
        print(f"\nEval loss at step {step}: {avg_loss:.4f}")
        return avg_loss

    def save_checkpoint(self, step, final=False, best=False):
        if best:
            checkpoint_dir = os.path.join(self.output_dir, "best")
        elif final:
            checkpoint_dir = os.path.join(self.output_dir, "final")
        else:
            checkpoint_dir = os.path.join(self.output_dir, f"checkpoint-{step}")
        os.makedirs(checkpoint_dir, exist_ok=True)

        model_to_save = (
            self.model.module if hasattr(self.model, "module") else self.model
        )

        if hasattr(model_to_save, "save_pretrained"):
            model_to_save.save_pretrained(checkpoint_dir)
        else:
            torch.save(
                model_to_save.state_dict(), os.path.join(checkpoint_dir, "model.pt")
            )

        torch.save(
            {
                "optimizer": self.optimizer.state_dict(),
                "scheduler": self.scheduler.state_dict(),
                "step": step,
            },
            os.path.join(checkpoint_dir, "training_state.pt"),
        )

        if self.tokenizer:
            self.tokenizer.save_pretrained(checkpoint_dir)

        print(f"Checkpoint saved: {checkpoint_dir}")


def create_train_val_split(dataset, val_ratio=0.05):
    val_size = int(len(dataset) * val_ratio)
    train_size = len(dataset) - val_size
    return random_split(dataset, [train_size, val_size])
