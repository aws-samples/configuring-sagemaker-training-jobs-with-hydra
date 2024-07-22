from pathlib import Path

from sklearn.datasets import fetch_openml

if __name__ == "__main__":
    Path("data").mkdir(parents=True, exist_ok=True)
    df = fetch_openml(data_id=41214, as_frame=True).frame
    df.to_csv("data/train.csv", index=False)
