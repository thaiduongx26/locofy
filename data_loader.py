from typing import Any, Dict

import torch
from torch.utils.data import Dataset


def convert_single_sample_to_tensor(sample: Dict[str, Any]):
    input, direction = sample["input"], sample["direction"]
    target = sample.get("target")
    left_space_input = torch.LongTensor([i[0] for i in input])
    size_input = torch.LongTensor([i[1] for i in input])
    direction_input = torch.LongTensor([direction] * len(input))
    # make sure the size of 3 inputs are equal
    assert (
        left_space_input.size() == size_input.size() == direction_input.size()
    ), f"Size of 3 inputs should be equal."
    if target:
        target = torch.LongTensor(target)
    return left_space_input, size_input, direction_input, target


class LocofyDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        sample = self.data[index]
        (
            left_space_input,
            size_input,
            direction_input,
            target,
        ) = convert_single_sample_to_tensor(sample)
        return left_space_input, size_input, direction_input, target
