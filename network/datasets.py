import os
from torch.utils.data import Dataset
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


class GraphDataset(torch.utils.data.Dataset):
    def __init__(self, input_dir, target_dir):
        super(GraphDataset, self).__init__()

        self.input_files = os.listdir(input_dir)
        self.input_files.sort()
        self.target_files = os.listdir(target_dir)
        self.target_files.sort()

        self.input_dir = input_dir
        self.target_dir = target_dir

    def __len__(self):
        return len(self.input_files)

    def __getitem__(self, idx):
        input_file = self.input_files[idx]
        target_file = self.target_files[idx]

        # Load the input graph
        input_graph = torch.load(os.path.join(self.input_dir, input_file))

        # Load the target graph
        target_graph = torch.load(os.path.join(self.target_dir, target_file))

        return input_graph, target_graph


