import yaml
from pathlib import Path

def test_load_config():
    cfg = yaml.safe_load(open("config/train.yaml", "r", encoding="utf-8"))
    assert "artifacts_dir" in cfg
    assert cfg["data"]["name"] == "iris"
    assert cfg["model"]["name"] == "logistic_regression"
    print("Config test passed.")

if __name__ == "__main__":
    test_load_config()