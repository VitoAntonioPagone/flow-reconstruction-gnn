import os
import torch
from torch.utils.data import Dataset
from datasets import CustomDataset

TRAIN_INPUT_DIR = '../dataset_graph/training/train_input_graphs'
TRAIN_TARGET_DIR = '../dataset_graph/training/train_graphs'

# Usage example:
train_ds = CustomDataset(TRAIN_INPUT_DIR, TRAIN_TARGET_DIR)

input_graph, label_graph = train_ds[0]
print(input_graph)
print(label_graph)
