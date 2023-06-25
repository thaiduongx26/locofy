import json
from typing import Any, Dict, List

import numpy as np
from tqdm import tqdm


def read_raw_data(path):
    """
    Read raw data from json file.
    """
    with open(path, "r") as f:
        data = json.load(f)
    return data


def scale_size(size_list: List[int], max_size: int):
    """
    Scale size to make sure it always less than max_size.
    """
    if max(size_list) <= max_size:
        return size_list
    return np.array(size_list) * max_size // max(size_list)


def truncate_list(lst, size):
    """
    Truncate a list into smaller lists with size.
    """
    truncated_lists = []
    for i in range(0, len(lst), size):
        truncated_lists.append(lst[i : i + size])
    return truncated_lists


def process_sample(
    sample: Dict[str, Any],
    max_size: int,
    max_length: int,
    contain_label=True,
    X_padding=0,
    y_padding=-1,
):
    """
    Process a sample from raw data. Apply processing steps: truncate, scaling size, encode the label
    """
    # encode label 0 for start of group, 1 for inside of group
    encoded_label: List[int] = []
    input_left_spaces: List[float] = []
    input_sizes: List[float] = []
    input = sample["input"]
    direction = sample["direction"]

    # scale size
    object_names = [i[0] for i in input if i[0] != "0"]
    objects = [i[0] for i in input]
    sizes = [i[1] for i in input]
    scale_sizes = scale_size(sizes, max_size)
    new_input = list(zip(objects, scale_sizes))
    for i in range(len(new_input)):
        if new_input[i][0] != "0":
            if i > 0 and new_input[i - 1][0] == "0":
                input_left_spaces.append(new_input[i - 1][1])
            else:
                input_left_spaces.append(0)
            input_sizes.append(new_input[i][1])
    new_input = list(zip(input_left_spaces, input_sizes))

    # encode direction
    direction = ["horizontal", "vertical"].index(direction)

    # encode label
    if contain_label:
        target = sample["output"]
        for i in range(len(target)):
            if isinstance(target[i], list):
                for j in range(len(target[i])):
                    encoded_label.append(min(1, j))
            else:
                encoded_label.append(0)

        # length of number of objects should be equal to length of label
        assert len(new_input) == len(
            encoded_label
        ), f"Length of input ({len(new_input)}) and label ({len(encoded_label)}) should be equal."
        assert len(object_names) == len(
            encoded_label
        ), f"Length of objects ({len(object_names)}) and label ({len(encoded_label)}) should be equal."

    # truncate
    truncated_data = []
    truncated_inputs = truncate_list(new_input, max_length)
    truncated_encoded_labels = truncate_list(encoded_label, max_length)
    for chunk in range(len(truncated_inputs)):
        # apply padding for each chunk
        chunk_size_padding: List[List[float]] = [(X_padding, X_padding)] * max_length
        chunk_size_padding[: len(truncated_inputs[chunk])] = truncated_inputs[chunk]
        chunk_label_padding: List[int] = [y_padding] * max_length
        if contain_label:
            chunk_label_padding[
                : len(truncated_encoded_labels[chunk])
            ] = truncated_encoded_labels[chunk]
        truncated_data.append(
            {
                "input": chunk_size_padding,
                "object_names": object_names,
                "direction": direction,
                "target": chunk_label_padding,
            }
        )

    return truncated_data


def process_raw_data(
    data: List[Dict[str, Any]],
    max_size: int,
    max_length: int,
    X_padding=0,
    y_padding=-1,
):
    """
    Apply some preprocessing steps to all data.
    """
    processed_data = []

    for sample in tqdm(data):
        processed_data.extend(
            process_sample(
                sample, max_size, max_length, X_padding=X_padding, y_padding=y_padding
            )
        )
    return processed_data


class DtypeEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        return json.JSONEncoder.default(self, obj)
