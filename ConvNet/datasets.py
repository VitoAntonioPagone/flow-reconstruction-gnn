import os
import re
import numpy as np
import torch
from torch.utils.data import Dataset as TorchDataset
import random


class FlowDataset(TorchDataset):
    """Dataset for flow field data with missing regions."""

    def __init__(self, input_dir, label_dir, add_mask=False):
        self.input_dir = input_dir
        self.label_dir = label_dir
        self.labels = sorted(os.listdir(label_dir))
        self.inputs = [re.sub("_label.npy$", "_input.npy", f) for f in self.labels]
        self.add_mask = add_mask

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, index):
        input_path = os.path.join(self.input_dir, self.inputs[index])
        label_path = os.path.join(self.label_dir, self.labels[index])

        input_data = np.load(input_path)
        label_data = np.load(label_path)

        input_tensor = torch.from_numpy(input_data).permute(2, 0, 1).float()
        label_tensor = torch.from_numpy(label_data).permute(2, 0, 1).float()

        missing_mask_tensor = (input_tensor[0] == 0).unsqueeze(0).float()

        if self.add_mask:
            input_tensor = torch.cat((input_tensor, missing_mask_tensor), dim=0)

        return input_tensor, label_tensor, missing_mask_tensor


class CustomDataset(torch.utils.data.Dataset):
    """Dataset for graph input-label pairs with subset selection."""

    def __init__(self, inputs_dir, labels_dir, subset_ratio=0.2):
        self.inputs_dir = inputs_dir
        self.labels_dir = labels_dir

        all_input_files = sorted(
            [file for file in os.listdir(self.inputs_dir) if file.endswith("_input.pt")]
        )

        num_files = int(len(all_input_files) * subset_ratio)

        self.input_files = random.sample(all_input_files, num_files)

    def __len__(self):
        return len(self.input_files)

    def __getitem__(self, idx):
        input_data = torch.load(os.path.join(self.inputs_dir, self.input_files[idx]))
        label_filename = self.input_files[idx].replace("_input.pt", "_label.pt")
        label_data = torch.load(os.path.join(self.labels_dir, label_filename))

        input_data.y = label_data.x
        input_data.x_complete = label_data.x

        return input_data
