from src.config import load_config

def test_config_loads():
    cfg = load_config()
    assert "model" in cfg and "train" in cfg and "paths" in cfg