from sklearn.linear_model import LogisticRegression

def build_model(cfg: dict):
    name = cfg["model"].get("name", "logistic_regression")

    if name != "logistic_regression":
        raise ValueError(f"Model not supported: {name}")

    return LogisticRegression(max_iter=int(cfg["model"]["max_iter"]))
