# src/evaluation/metrics.py
from __future__ import annotations
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, classification_report

LABELS = ["negative", "neutral", "positive"]  # consistent ordering

label_to_id = {l: i for i, l in enumerate(LABELS)}
id_to_label = {i: l for i, l in enumerate(LABELS)}

def compute_metrics(y_true_ids, y_pred_ids):
    acc = accuracy_score(y_true_ids, y_pred_ids)
    f1_macro = f1_score(y_true_ids, y_pred_ids, average="macro")
    f1_weighted = f1_score(y_true_ids, y_pred_ids, average="weighted")
    return {"accuracy": acc, "f1_macro": f1_macro, "f1_weighted": f1_weighted}

def classification_report_str(y_true_ids, y_pred_ids) -> str:
    return classification_report(
        y_true_ids, y_pred_ids,
        target_names=LABELS, digits=4
    )