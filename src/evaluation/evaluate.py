# src/evaluation/evaluate.py
from __future__ import annotations
import argparse
from pathlib import Path
import math
import json

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

from transformers import AutoTokenizer
from src.config import load_config
from src.training.utils import get_device, set_seed, load_checkpoint
from src.models.modeling import BertForSentiment
from src.evaluation.metrics import LABELS, label_to_id, id_to_label, compute_metrics, classification_report_str


def _ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

@torch.no_grad()
def _predict_batch(texts, tokenizer, model, device, max_length: int):
    enc = tokenizer(
        list(texts),
        truncation=True,
        padding=True,
        max_length=max_length,
        return_tensors="pt",
    )
    enc = {k: v.to(device) for k, v in enc.items()}
    out = model(**enc)
    probs = torch.softmax(out["logits"], dim=-1).detach().cpu().numpy()
    pred_ids = probs.argmax(axis=-1)
    return probs, pred_ids

def _load_trained(model_dir: Path):
    tokenizer = AutoTokenizer.from_pretrained(model_dir, use_fast=True)
    model = load_checkpoint(BertForSentiment, model_dir)
    model.eval()
    device = get_device()
    model.to(device)
    return model, tokenizer, device

def _save_confusion_matrix(y_true, y_pred, out_png: Path, title: str):
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(LABELS))))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=LABELS)
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    disp.plot(ax=ax, xticks_rotation=45, colorbar=False)
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(out_png, dpi=160)
    plt.close(fig)

def eval_ai_test(cfg):
    paths = cfg["paths"]
    out_root = Path(paths["outputs_dir"]) / "eval"
    _ensure_dir(out_root)

    model_dir = Path(paths["model_dir"]) / "best"
    model, tokenizer, device = _load_trained(model_dir)
    max_len = int(cfg["model"]["max_length"])

    # Load AI test split
    test_csv = Path(paths["outputs_dir"]) / "splits" / "test.csv"
    df = pd.read_csv(test_csv)

    texts = df["text_clean"].astype(str).tolist()
    y_true = df[cfg["data"]["label_col"]].astype(str).str.lower().map(label_to_id).tolist()

    # Batched predictions (chunked for memory safety on CPU)
    batch_size = 32
    all_probs, all_preds = [], []
    for i in range(0, len(texts), batch_size):
        probs, pred_ids = _predict_batch(texts[i:i+batch_size], tokenizer, model, device, max_len)
        all_probs.append(probs)
        all_preds.append(pred_ids)
    probs = np.vstack(all_probs)
    y_pred = np.concatenate(all_preds)

    # Metrics & reports
    metrics = compute_metrics(y_true, y_pred)
    report = classification_report_str(y_true, y_pred)
    (out_root / "ai_test_report.txt").write_text(report, encoding="utf-8")
    (out_root / "ai_test_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    # Confusion matrix
    _save_confusion_matrix(y_true, y_pred, out_root / "ai_test_confusion_matrix.png", "AI Test — Confusion Matrix")

    # Detailed predictions
    pred_labels = [id_to_label[i] for i in y_pred]
    confs = probs.max(axis=1)
    out_df = df.copy()
    out_df["pred"] = pred_labels
    out_df["confidence"] = confs
    for i, lab in enumerate(LABELS):
        out_df[f"p_{lab}"] = probs[:, i]
    out_df.to_csv(out_root / "ai_test_predictions.csv", index=False, encoding="utf-8")

    # Error analysis (only wrong predictions), top-N most confident mistakes
    errors = out_df[out_df[cfg["data"]["label_col"]] != out_df["pred"]].copy()
    errors = errors.sort_values("confidence", ascending=False)
    errors_cols = ["text_clean", "language", cfg["data"]["label_col"], "pred", "confidence", "p_negative", "p_neutral", "p_positive", "__source_file"]
    errors_cols = [c for c in errors_cols if c in errors.columns]
    errors[errors_cols].to_csv(out_root / "ai_test_errors_top.csv", index=False, encoding="utf-8")

    print("\n=== AI Test Results ===")
    print(metrics)
    print("\nClassification report saved to outputs/eval/ai_test_report.txt")
    print("Predictions  -> outputs/eval/ai_test_predictions.csv")
    print("Errors (top) -> outputs/eval/ai_test_errors_top.csv")
    print("Confusion matrix -> outputs/eval/ai_test_confusion_matrix.png")

def eval_real(cfg):
    paths = cfg["paths"]
    out_root = Path(paths["outputs_dir"]) / "eval"
    _ensure_dir(out_root)

    model_dir = Path(paths["model_dir"]) / "best"
    model, tokenizer, device = _load_trained(model_dir)
    max_len = int(cfg["model"]["max_length"])

    # Real data
    real_csv = Path(cfg["data"]["real_path"]).resolve()
    if not real_csv.exists():
        raise FileNotFoundError(f"Real reviews CSV not found: {real_csv}")
    df = pd.read_csv(real_csv)

    # Flexible: if text_clean missing, fall back to text then run minimal cleaning assumptions
    text_col = "text_clean" if "text_clean" in df.columns else cfg["data"]["text_col"]
    if text_col not in df.columns:
        raise ValueError(f"Expected '{text_col}' or '{cfg['data']['text_col']}' in real CSV.")

    texts = df[text_col].astype(str).tolist()

    # Predict
    batch_size = 32
    all_probs, all_preds = [], []
    for i in range(0, len(texts), batch_size):
        probs, pred_ids = _predict_batch(texts[i:i+batch_size], tokenizer, model, device, max_len)
        all_probs.append(probs)
        all_preds.append(pred_ids)
    probs = np.vstack(all_probs)
    y_pred = np.concatenate(all_preds)
    pred_labels = [id_to_label[i] for i in y_pred]
    confs = probs.max(axis=1)

    out_df = df.copy()
    out_df["pred"] = pred_labels
    out_df["confidence"] = confs
    for i, lab in enumerate(LABELS):
        out_df[f"p_{lab}"] = probs[:, i]

    # If real labels provided, compute metrics; else just save predictions
    label_col = cfg["data"]["label_col"]
    if label_col in df.columns:
        y_true = df[label_col].astype(str).str.lower().map(label_to_id)
        if y_true.isnull().any():
            print("[warn] Some real labels are invalid; metrics will ignore them.")
        valid_mask = ~y_true.isnull()
        y_true_ids = y_true[valid_mask].astype(int).values
        y_pred_ids = y_pred[valid_mask]
        metrics = compute_metrics(y_true_ids, y_pred_ids)
        report = classification_report_str(y_true_ids, y_pred_ids)
        (out_root / "real_test_report.txt").write_text(report, encoding="utf-8")
        (out_root / "real_test_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
        _save_confusion_matrix(y_true_ids, y_pred_ids, out_root / "real_test_confusion_matrix.png", "Real Test — Confusion Matrix")
        print("\n=== Real Data Results (with labels) ===")
        print(metrics)
        print("Report -> outputs/eval/real_test_report.txt")
        print("Confusion matrix -> outputs/eval/real_test_confusion_matrix.png")
    else:
        print("\n[info] Real CSV has no ground-truth labels; saving predictions only.")

    out_df.to_csv(out_root / "real_predictions.csv", index=False, encoding="utf-8")
    print("Predictions -> outputs/eval/real_predictions.csv")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--which", choices=["ai", "real", "both"], default="both",
                        help="Evaluate on AI test split, real dataset, or both.")
    args = parser.parse_args()

    cfg = load_config()
    set_seed(cfg["train"]["seed"])

    if args.which in ("ai", "both"):
        eval_ai_test(cfg)
    if args.which in ("real", "both"):
        eval_real(cfg)

if __name__ == "__main__":
    main()