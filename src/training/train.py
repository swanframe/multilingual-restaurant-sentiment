# src/training/train.py
from __future__ import annotations
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, DataCollatorWithPadding, get_linear_schedule_with_warmup

from src.config import load_config
from src.training.utils import set_seed, get_device, save_checkpoint
from src.models.modeling import BertForSentiment
from src.evaluation.metrics import label_to_id, id_to_label, compute_metrics, classification_report_str

# ---------- Dataset ----------
class ReviewsDataset(Dataset):
    def __init__(self, csv_path: str, tokenizer, text_col: str, lang_col: str, label_col: str, max_length: int):
        df = pd.read_csv(csv_path)
        # We rely on Phase 2 outputs: text_clean already exists
        self.texts = df["text_clean"].astype(str).tolist()
        self.langs = df[lang_col].astype(str).tolist()
        labels = df[label_col].astype(str).str.lower().map(label_to_id)
        self.labels = labels.tolist()
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        # We could prefix with language tag if desired; often not needed for mBERT.
        enc = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            return_tensors=None,
            padding=False
        )
        return {
            "input_ids": torch.tensor(enc["input_ids"], dtype=torch.long),
            "attention_mask": torch.tensor(enc["attention_mask"], dtype=torch.long),
            # token_type_ids may be absent for some models; add if present
            "labels": torch.tensor(self.labels[idx], dtype=torch.long),
        }

# custom collator to handle token_type_ids if tokenizer returns it
@dataclass
class CollateWithTokenizer:
    tokenizer: AutoTokenizer

    def __call__(self, batch):
        # batch is list of dicts with lists in "input_ids"/"attention_mask"
        keys = batch[0].keys()
        features = {k: [ex[k] for ex in batch] for k in keys}
        # Use transformers' DataCollatorWithPadding for convenience
        data_collator = DataCollatorWithPadding(self.tokenizer, padding=True)
        features_for_collator = [{"input_ids": f.tolist() if torch.is_tensor(f) else f,
                                  "attention_mask": a.tolist() if torch.is_tensor(a) else a,
                                  **({"token_type_ids": []} if False else {})}
                                 for f, a in zip(features["input_ids"], features["attention_mask"])]
        # Manually add labels after padding
        out = data_collator(features_for_collator)
        out["labels"] = torch.stack(features["labels"])
        return out

# ---------- Training loop ----------
def train_one_epoch(model, loader, optimizer, device, scheduler=None, grad_clip: float | None = 1.0):
    model.train()
    total_loss = 0.0
    for step, batch in enumerate(loader):
        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = model(**batch)
        loss = outputs["loss"]
        loss.backward()

        if grad_clip is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

        optimizer.step()
        if scheduler:
            scheduler.step()
        optimizer.zero_grad(set_to_none=True)
        total_loss += loss.item()
    return total_loss / max(1, len(loader))

@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    losses = []
    preds, trues = [], []
    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = model(**batch)
        if outputs["loss"] is not None:
            losses.append(outputs["loss"].item())
        logits = outputs["logits"]
        pred_ids = torch.argmax(logits, dim=-1).detach().cpu().tolist()
        true_ids = batch["labels"].detach().cpu().tolist()
        preds.extend(pred_ids)
        trues.extend(true_ids)
    metrics = compute_metrics(trues, preds)
    loss = float(sum(losses) / max(1, len(losses))) if losses else math.nan
    return loss, metrics, trues, preds

def main():
    cfg = load_config()
    set_seed(cfg["train"]["seed"])
    device = get_device()

    model_name = cfg["model"]["pretrained_name"]
    max_len = int(cfg["model"]["max_length"])
    batch_size = int(cfg["train"]["batch_size"])
    epochs = int(cfg["train"]["epochs"])
    lr = float(cfg["train"]["lr"])
    weight_decay = float(cfg["train"]["weight_decay"])

    paths = cfg["paths"]
    out_dir = Path(paths["model_dir"]) / "best"
    logs_dir = Path(paths["logs_dir"])
    logs_dir.mkdir(parents=True, exist_ok=True)

    # Tokenizer (WordPiece; no sentencepiece)
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)

    # Datasets
    splits_dir = Path(paths["outputs_dir"]) / "splits"
    train_ds = ReviewsDataset(splits_dir / "train.csv", tokenizer,
                              text_col=cfg["data"]["text_col"],
                              lang_col=cfg["data"]["lang_col"],
                              label_col=cfg["data"]["label_col"],
                              max_length=max_len)
    val_ds = ReviewsDataset(splits_dir / "val.csv", tokenizer,
                            text_col=cfg["data"]["text_col"],
                            lang_col=cfg["data"]["lang_col"],
                            label_col=cfg["data"]["label_col"],
                            max_length=max_len)

    collate = CollateWithTokenizer(tokenizer)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=cfg["train"]["num_workers"], collate_fn=collate)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=cfg["train"]["num_workers"], collate_fn=collate)

    # Model
    model = BertForSentiment(pretrained_name=model_name, num_labels=3, dropout=cfg["model"]["dropout"])
    model.to(device)

    # Optimizer & Scheduler
    no_decay = ["bias", "LayerNorm.weight"]
    grouped = [
        {"params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
         "weight_decay": weight_decay},
        {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
         "weight_decay": 0.0},
    ]
    optimizer = torch.optim.AdamW(grouped, lr=lr)

    total_steps = epochs * len(train_loader)
    warmup_steps = int(0.1 * total_steps)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps
    )

    best_f1 = -1.0
    patience = 2
    bad_epochs = 0

    for epoch in range(1, epochs + 1):
        t0 = time.time()
        train_loss = train_one_epoch(model, train_loader, optimizer, device, scheduler=scheduler, grad_clip=1.0)
        val_loss, val_metrics, y_true, y_pred = evaluate(model, val_loader, device)
        dur = time.time() - t0

        msg = (f"Epoch {epoch}/{epochs} | "
               f"train_loss={train_loss:.4f} | "
               f"val_loss={val_loss:.4f} | "
               f"val_acc={val_metrics['accuracy']:.4f} | "
               f"val_f1_macro={val_metrics['f1_macro']:.4f} | "
               f"time={dur:.1f}s")
        print(msg)

        # Early stopping on macro-F1
        if val_metrics["f1_macro"] > best_f1:
            best_f1 = val_metrics["f1_macro"]
            bad_epochs = 0
            save_checkpoint(
                model, tokenizer, out_dir,
                extra={"best_f1_macro": best_f1, "epoch": epoch}
            )
            print(f"  ✓ Saved new best to {out_dir} (f1_macro={best_f1:.4f})")
            # also write a quick report
            (Path(paths["outputs_dir"]) / "val_report.txt").write_text(
                classification_report_str(y_true, y_pred),
                encoding="utf-8"
            )
        else:
            bad_epochs += 1
            if bad_epochs > patience:
                print("  Early stopping triggered.")
                break

    print("\nTraining complete.")
    print(f"Best macro-F1: {best_f1:.4f} | model dir: {out_dir}")

if __name__ == "__main__":
    main()