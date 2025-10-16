# src/inference/predict.py
from __future__ import annotations
from pathlib import Path
import torch
from transformers import AutoTokenizer
from src.models.modeling import BertForSentiment
from src.evaluation.metrics import id_to_label
from src.training.utils import load_checkpoint  # Use our updated loader

def load_model(model_dir: str | Path):
    model_dir = Path(model_dir)
    tokenizer = AutoTokenizer.from_pretrained(model_dir, use_fast=True)
    model = load_checkpoint(BertForSentiment, model_dir)
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    return model, tokenizer, device

@torch.no_grad()
def predict_one(text: str, model, tokenizer, device, max_length: int = 192):
    enc = tokenizer(text, truncation=True, max_length=max_length, return_tensors="pt")
    enc = {k: v.to(device) for k, v in enc.items()}
    out = model(**enc)
    probs = torch.softmax(out["logits"], dim=-1)[0]
    pred_id = int(torch.argmax(probs).item())
    return id_to_label[pred_id], float(probs[pred_id].item())