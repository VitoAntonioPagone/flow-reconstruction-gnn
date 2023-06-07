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


class GraphDataset(Dataset):
    def __init__(self, root_dir, transform=None, pre_transform=None):
        super(GraphDataset, self).__init__(root_dir, transform, pre_transform)

    @property
    def raw_file_names(self):
        # List all files in the raw_dir
        return os.listdir(self.raw_dir)

    @property
    def processed_file_names(self):
        # Once processed, file names should follow this format
        return [f"data_{i}.pt" for i in range(len(self.raw_paths))]
    
    def process(self):
        # Process files one by one
        for raw_path in self.raw_paths:
            # Load a raw file
            data = torch.load(raw_path)
            
            # Process the data into suitable format and save it
            torch.save(data, os.path.join(self.processed_dir, f"data_{os.path.splitext(os.path.basename(raw_path))[0]}.pt"))

    def len(self):
        # The length of the dataset is simply the number of processed files
        return len(self.processed_file_names)

    def get(self, idx):
        # Load a processed file with the given index
        data = torch.load(os.path.join(self.processed_dir, f"data_{idx}.pt"))
        return data
