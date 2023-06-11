import os
from torch_geometric.data import Dataset, Data
import torch
import numpy as np
import re

class FlowDataset(Dataset):
    def __init__(self, input_dir, label_dir, add_mask=False):
        self.input_dir = input_dir
        self.label_dir = label_dir
        self.labels = sorted(os.listdir(label_dir))
        self.inputs = [re.sub('_label.npy$', '_input.npy', f) for f in self.labels ]  
        self.add_mask = add_mask

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, index):
        input_path = os.path.join(self.input_dir, self.inputs[index])
        label_path = os.path.join(self.label_dir, self.labels[index])

        input_data = np.load(input_path)
        label_data = np.load(label_path)

        # Convert the numpy arrays to PyTorch tensors and add the channel dimension
        input_tensor = torch.from_numpy(input_data).permute(2, 0, 1).float()
        label_tensor = torch.from_numpy(label_data).permute(2, 0, 1).float()

        # Compute the single-channel missing mask tensor
        missing_mask_tensor = (input_tensor[0] == 0).unsqueeze(0).float()

        if self.add_mask:
            input_tensor = torch.cat((input_tensor, missing_mask_tensor), dim=0)

        return input_tensor, label_tensor, missing_mask_tensor


import os
import torch
from torch.utils.data import Dataset

class CustomDataset(Dataset):
    def __init__(self, inputs_dir, labels_dir):
        self.inputs_dir = inputs_dir
        self.labels_dir = labels_dir
        self.input_files = sorted([file for file in os.listdir(self.inputs_dir) if file.endswith('_input.pt')])

    def __len__(self):
        return len(self.input_files)

    def __getitem__(self, idx):
        input_data = torch.load(os.path.join(self.inputs_dir, self.input_files[idx]))
        label_filename = self.input_files[idx].replace('_input.pt', '_label.pt')
        label_data = torch.load(os.path.join(self.labels_dir, label_filename))

        return input_data, label_data