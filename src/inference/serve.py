# src/inference/serve.py
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple
import time
import csv
import threading

import numpy as np
import torch
from transformers import AutoTokenizer

from src.config import load_config
from src.training.utils import get_device, load_checkpoint
from src.models.modeling import BertForSentiment
from src.evaluation.metrics import LABELS, id_to_label


@dataclass
class Thresholds:
    general: float
    per_class: Dict[str, float]

    def flag(self, label: str, conf: float) -> bool:
        t = self.per_class.get(label, self.general)
        return conf < t


class Predictor:
    """Thread-safe, eager-loaded predictor for Flask."""
    def __init__(self, cfg: Dict):
        paths = cfg["paths"]
        serve = cfg["serve"]
        model_dir = Path(paths["model_dir"]) / serve["model_subdir"]
        self.max_len = int(serve["max_length"])

        self.model = load_checkpoint(BertForSentiment, model_dir)
        self.model.eval()
        self.device = get_device()
        self.model.to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir, use_fast=True)

        self.thresholds = Thresholds(
            general=float(serve["general_threshold"]),
            per_class={k: float(v) for k, v in serve["per_class_thresholds"].items()},
        )

        # Logging setup
        self.api_dir = Path(paths["outputs_dir"]) / "api_logs"
        self.pred_csv = Path(serve["logs"]["predictions_csv"])
        self.fb_csv = Path(serve["logs"]["feedback_csv"])
        for p in [self.api_dir, self.pred_csv.parent, self.fb_csv.parent]:
            p.mkdir(parents=True, exist_ok=True)

        # Write CSV headers if empty
        if not self.pred_csv.exists():
            with open(self.pred_csv, "w", newline="", encoding="utf-8") as f:
                w = csv.writer(f)
                w.writerow(["ts", "text", "language", "pred", "confidence",
                            "p_negative", "p_neutral", "p_positive", "low_confidence",
                            "advisory"])
        if not self.fb_csv.exists():
            with open(self.fb_csv, "w", newline="", encoding="utf-8") as f:
                w = csv.writer(f)
                w.writerow(["ts", "text", "language", "pred", "confidence", "true_label", "notes"])

        self._lock = threading.Lock()

    @torch.no_grad()
    def predict_batch(self, texts: List[str]) -> Tuple[np.ndarray, np.ndarray]:
        enc = self.tokenizer(
            texts, truncation=True, padding=True,
            max_length=self.max_len, return_tensors="pt"
        )
        enc = {k: v.to(self.device) for k, v in enc.items()}
        out = self.model(**enc)
        probs = torch.softmax(out["logits"], dim=-1).detach().cpu().numpy()
        pred_ids = probs.argmax(axis=-1)
        return probs, pred_ids

    def advise(self, label: str, conf: float) -> Tuple[bool, str]:
        low = self.thresholds.flag(label, conf)
        if not low:
            return False, ""
        # Human-review advisory tailored for minority sentiments
        if label in ("negative", "neutral"):
            return True, "Low confidence on minority sentiment—consider human review."
        return True, "Low confidence—consider a second opinion or more context."

    def log_prediction(self, row: Dict):
        with self._lock:
            with open(self.pred_csv, "a", newline="", encoding="utf-8") as f:
                w = csv.writer(f)
                w.writerow([row.get("ts"), row.get("text"), row.get("language"),
                            row.get("pred"), row.get("confidence"),
                            row.get("p_negative"), row.get("p_neutral"), row.get("p_positive"),
                            row.get("low_confidence"), row.get("advisory")])

    def log_feedback(self, row: Dict):
        with self._lock:
            with open(self.fb_csv, "a", newline="", encoding="utf-8") as f:
                w = csv.writer(f)
                w.writerow([row.get("ts"), row.get("text"), row.get("language"),
                            row.get("pred"), row.get("confidence"), row.get("true_label"),
                            row.get("notes", "")])

    def predict_payload(self, items: List[Dict]) -> List[Dict]:
        texts = [str(x.get("text", "")) for x in items]
        probs, pred_ids = self.predict_batch(texts)
        ts = int(time.time())

        results: List[Dict] = []
        for i, pid in enumerate(pred_ids):
            label = id_to_label[int(pid)]
            conf = float(probs[i, pid])
            low, advisory = self.advise(label, conf)
            row = {
                "ts": ts,
                "text": texts[i],
                "language": (items[i].get("language") or "").lower()[:5],
                "pred": label,
                "confidence": conf,
                "p_negative": float(probs[i, 0]),
                "p_neutral": float(probs[i, 1]),
                "p_positive": float(probs[i, 2]),
                "low_confidence": low,
                "advisory": advisory
            }
            results.append(row)
            self.log_prediction(row)
        return results