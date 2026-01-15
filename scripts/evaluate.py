import json
from pathlib import Path
import joblib
import yaml

def load_cfg(path="config/train.yaml"):
    return yaml.safe_load(open(path, "r", encoding="utf-8"))

def main():
    cfg = load_cfg()
    art_dir = Path(cfg.get("artifacts_dir", "artifacts"))
    model_path = art_dir / "model.joblib"
    metrics_path = art_dir / "metrics.json"

    if not model_path.exists():
        print("Model not found. Run train.py first.")
        return

    pipe = joblib.load(model_path)
    metrics = json.load(open(metrics_path, "r"))
    print("Loaded model metrics:", metrics)

    # For evaluation, perhaps load test data, but since no separate, just print
    print("Evaluation complete.")

if __name__ == "__main__":
    main()