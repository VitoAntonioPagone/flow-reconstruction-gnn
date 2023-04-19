import torch
import numpy as np
from torch.utils.data import Dataset

class FlowDataset(Dataset):
    def __init__(self, train_files, label_files):
        self.train_files = train_files
        self.label_files = label_files

    def __len__(self):
        return len(self.train_files)

    def __getitem__(self, idx):
        train_data = np.load(self.train_files[idx])
        label_data = np.load(self.label_files[idx])

        # Convert the numpy arrays to PyTorch tensors and add the channel dimension
        train_tensor = torch.from_numpy(train_data).permute(2, 0, 1).float()
        label_tensor = torch.from_numpy(label_data).permute(2, 0, 1).float()

        # Compute the single-channel missing mask tensor
        missing_mask_tensor = (train_tensor[0] == 0).unsqueeze(0).float()

        return train_tensor, label_tensor, missing_mask_tensor

