import json
import random
from glob import glob
from typing import Any, Dict, List

from tqdm import tqdm

from utils import read_raw_data


def kfold_split(data: List[Dict[str, Any]], kfold=5, shuffle=True):
    """
    Split data into k folds.
    """
    if shuffle:
        random.shuffle(data)

    n_test_samples = len(data) // kfold

    for i in tqdm(range(kfold)):
        test_data = data[i * n_test_samples : (i + 1) * n_test_samples]
        train_data = data[: i * n_test_samples] + data[(i + 1) * n_test_samples :]
        # save fold i to json file
        with open(f"./Data/fold_{i}.json", "w") as f:
            json.dump({"train": train_data, "test": test_data}, f)

    # save another version of all data to json file
    with open(f"./Data/all.json", "w") as f:
        json.dump({"train": data}, f)


if __name__ == "__main__":
    data_path = "./Data/"
    files = glob(data_path + "*.json")
    data = []
    for file in files:
        content = read_raw_data(file)
        data.extend(content)
    kfold_split(data, kfold=5, shuffle=True)
