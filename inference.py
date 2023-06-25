import glob
import logging
from pathlib import Path
from typing import Any, Dict, List, Union

import torch

from data_loader import convert_single_sample_to_tensor
from model import ModelConfig, SequenceLabelingModel
from utils import process_sample

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


class ServingModel:
    def __init__(self, model_paths: List[Path] = None):
        super(ServingModel, self).__init__()
        self.model_config = ModelConfig()
        self.models: List[SequenceLabelingModel] = []
        if model_paths is None:
            model_paths = glob.glob("Output/trained_model*.pt")
        self.load_model(model_paths)

    def load_model(self, model_paths: List[Path]):
        for path in model_paths:
            model = SequenceLabelingModel(
                self.model_config.encoder_hidden_size, self.model_config.num_labels
            )
            model.load_state_dict(torch.load(path))
            model.eval()
            self.models.append(model)

    def single_predict(self, input: Dict[str, Any]) -> List[Union[List[str], str]]:
        prediction_output: List[Union[List[str], str]] = []
        processed_input = process_sample(
            input,
            max_size=self.model_config.max_size_value,
            max_length=self.model_config.max_sequence_length,
            contain_label=False,
        )

        # name of objects like 'a', 'b', 'c', ... following original order of input.
        object_names: List[str] = []
        encoded_output: List[int] = []

        for chunk in processed_input:
            (
                left_space_input,
                size_input,
                direction_input,
                _,
            ) = convert_single_sample_to_tensor(chunk)
            object_names.extend(chunk["object_names"])
            left_space_input = left_space_input.reshape(1, -1)
            size_input = size_input.reshape(1, -1)
            direction_input = direction_input.reshape(1, -1)
            output = self.models_forward(left_space_input, size_input, direction_input)
            output = output.reshape(-1).tolist()[: len(object_names)]
            encoded_output.extend(output)

        # Convert encoded output to correct output format.
        group_elements: List[str] = []  # contain label in a group
        index = 0
        assert len(encoded_output) == len(
            object_names
        ), f"Length of output ({len(encoded_output)}) and object names ({len(object_names)}) should be equal."
        while index < len(encoded_output):
            if encoded_output[index] == 0:
                if len(group_elements):
                    if len(group_elements) == 1:
                        prediction_output.append(group_elements[0])
                    else:
                        prediction_output.append(group_elements)
                group_elements = []
            group_elements.append(object_names[index])
            index += 1
        if len(group_elements):
            if len(group_elements) == 1:
                prediction_output.append(group_elements[0])
            else:
                prediction_output.append(group_elements)
        return prediction_output

    def models_forward(
        self,
        left_space_input: torch.Tensor,
        size_input: torch.Tensor,
        direction_input: torch.Tensor,
    ):
        """Concat the outputs of all models and ensenble them using average.

        Args:
            left_space_input (torch.Tensor): _description_
            size_input (torch.Tensor): _description_
            direction_input (torch.Tensor): _description_

        Returns:
            _type_: _description_
        """
        outputs = []
        for model in self.models:
            output = model(left_space_input, size_input, direction_input)
            outputs.append(output)

        outputs = torch.stack(outputs, dim=0)
        outputs = torch.mean(outputs, dim=0)
        outputs = torch.argmax(outputs, dim=2)
        return outputs


if __name__ == "__main__":
    serve = ServingModel()
    logger.debug(
        serve.single_predict(
            {
                "id": 17,
                "input": [
                    ["a", 180],
                    ["0", 12],
                    ["b", 30],
                    ["0", 4],
                    ["c", 72],
                    ["0", 8],
                    ["d", 24],
                    ["0", 24],
                ],
                "output": ["a", ["b", "c", "d"]],
                "direction": "vertical",
            },
        )
    )
