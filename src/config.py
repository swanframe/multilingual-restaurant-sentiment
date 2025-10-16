from pathlib import Path
import os
import yaml
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = PROJECT_ROOT / "configs" / "config.yaml"

def load_config():
    load_dotenv(PROJECT_ROOT / ".env", override=True)
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    # Expand to absolute paths
    for key in ["model_dir", "logs_dir", "outputs_dir"]:
        p = Path(cfg["paths"][key])
        cfg["paths"][key] = str((PROJECT_ROOT / p).resolve())
    return cfg