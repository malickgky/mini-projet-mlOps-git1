from sklearn.datasets import load_iris
import pandas as pd

def load_dataset(cfg: dict):
    data_name = cfg["data"]["name"]
    if data_name == "iris":
        data = load_iris()
        X = pd.DataFrame(data.data, columns=data.feature_names)
        y = pd.Series(data.target, name="target")
        return X, y
    else:
        raise ValueError(f"Dataset not supported: {data_name}")
