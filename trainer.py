import argparse
import json
import logging
import os
import random
import time
from datetime import datetime as time

import torch
import torch.nn as nn
import torch.optim as optim
from seqeval.metrics import classification_report
from torch.utils.data import DataLoader
from tqdm import tqdm

from data_loader import LocofyDataset
from model import ModelConfig, SequenceLabelingModel
from utils import DtypeEncoder, process_raw_data, read_raw_data

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


model_config = ModelConfig()


def create_parser():
    parser = argparse.ArgumentParser(description="Training arguments")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_epochs", type=int, default=20)
    parser.add_argument("--ver_name", type=str, default="fold_0")
    parser.add_argument("--output_dir", type=str, default="Output/")
    parser.add_argument("--do_train", action="store_true")
    parser.add_argument("--do_eval", action="store_true")
    parser.add_argument("--do_save", action="store_true")
    parser.add_argument("--seed", type=int, default=26)
    parser.add_argument("--checkpoint_path", type=str, default=None)
    args = parser.parse_args()
    return args


args = create_parser()
random.seed(args.seed)

data_path = f"Data/{args.ver_name}.json"
time_now = time.now().strftime("%Y-%m-%d-%H-%M-%S")

raw_data = read_raw_data(data_path)
train_data = process_raw_data(
    raw_data["train"],
    max_size=model_config.max_size_value,
    max_length=model_config.max_sequence_length,
)
train_dataset = LocofyDataset(train_data)
train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)

if args.do_eval:
    test_data = process_raw_data(
        raw_data["test"],
        max_size=model_config.max_size_value,
        max_length=model_config.max_sequence_length,
    )
    test_dataset = LocofyDataset(test_data)
    test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=False)

model = SequenceLabelingModel(model_config.encoder_hidden_size, model_config.num_labels)
if args.checkpoint_path:
    model.load_state_dict(torch.load(args.checkpoint_path))

optimizer = optim.Adam(model.parameters(), lr=model_config.learning_rate)
criterion = nn.CrossEntropyLoss(ignore_index=-1)

# Set the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)


def evaluate(model, dataloader):
    model.eval()
    label_map = {0: "B-GROUP", 1: "I-GROUP"}
    all_predictions = []
    all_labels = []
    for batch in tqdm(dataloader):
        left_space_input, size_input, direction_input, target = batch
        left_space_input = left_space_input.to(device)
        size_input = size_input.to(device)
        direction_input = direction_input.to(device)
        target = target.to(device)

        model_outputs = model(left_space_input, size_input, direction_input)
        for sample_index in range(target.shape[0]):
            # ignore padding index
            mask = target[sample_index] != -1
            # reshape the outputs and labels to match the mask
            outputs = model_outputs[sample_index][mask]
            outputs = torch.argmax(outputs, dim=1)
            outputs = [label_map[l.item()] for l in outputs]
            labels = target[sample_index][mask]
            labels = [label_map[l.item()] for l in labels]
            all_predictions.append(outputs)
            all_labels.append(labels)

    with open(
        os.path.join(args.output_dir, f"result_{args.ver_name}_{time_now}.json"), "w+"
    ) as f:
        json.dump(
            classification_report(all_labels, all_predictions, output_dict=True),
            f,
            cls=DtypeEncoder,
        )
    logger.info(classification_report(all_labels, all_predictions))


def train(model, dataloader, num_epochs=20):
    # training loop
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0

        for batch in tqdm(dataloader):
            left_space_input, size_input, direction_input, target = batch
            left_space_input = left_space_input.to(device)
            size_input = size_input.to(device)
            direction_input = direction_input.to(device)
            target = target.to(device)

            optimizer.zero_grad()
            outputs = model(left_space_input, size_input, direction_input)

            # ignore padding index
            mask = target != -1
            # reshape the outputs and labels to match the mask
            outputs = outputs[mask]
            labels = target[mask]
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        average_loss = total_loss / len(train_dataloader)
        logger.info(f"Epoch {epoch + 1}/{num_epochs}, Loss: {average_loss:.4f}")


if __name__ == "__main__":
    if args.do_train:
        train(model, train_dataloader, num_epochs=args.num_epochs)

    if args.do_eval:
        evaluate(model, test_dataloader)

    if args.do_save:
        torch.save(
            model.state_dict(),
            os.path.join(
                args.output_dir, f"trained_model_{args.ver_name}_{time_now}.pt"
            ),
        )
