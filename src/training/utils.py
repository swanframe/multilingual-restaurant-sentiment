# src/training/utils.py
from __future__ import annotations
import os
import random
import numpy as np
import torch
import json
from pathlib import Path
from typing import Dict
from src.utils.io import ensure_dir

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")

def save_checkpoint(model, tokenizer, out_dir: str | os.PathLike, extra: Dict | None = None):
    out = Path(out_dir)
    ensure_dir(out)
    
    # Save model state dict instead of using save_pretrained
    torch.save(model.state_dict(), out / "pytorch_model.bin")
    
    # Save model configuration
    model_config = {
        "pretrained_name": model.pretrained_name if hasattr(model, 'pretrained_name') else "bert-base-multilingual-cased",
        "num_labels": model.classifier.out_features if hasattr(model, 'classifier') else 3,
        "dropout": model.dropout.p if hasattr(model, 'dropout') else 0.1
    }
    with open(out / "model_config.json", "w") as f:
        json.dump(model_config, f, indent=2)
    
    # Save tokenizer
    tokenizer.save_pretrained(out)
    
    if extra:
        torch.save(extra, out / "extra.pt")

def load_checkpoint(model_cls, out_dir: str | os.PathLike, **kwargs):
    out = Path(out_dir)
    
    # Load model configuration
    with open(out / "model_config.json", "r") as f:
        model_config = json.load(f)
    
    # Update with any provided kwargs
    model_config.update(kwargs)
    
    # Create model instance and load state dict
    model = model_cls(**model_config)
    model.load_state_dict(torch.load(out / "pytorch_model.bin", map_location="cpu"))
    return model