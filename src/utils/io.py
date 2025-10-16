from pathlib import Path

def ensure_dir(path: str | Path):
    Path(path).mkdir(parents=True, exist_ok=True)