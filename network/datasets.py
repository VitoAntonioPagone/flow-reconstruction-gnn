import os
from torch.utils.data import Dataset
import torch
import numpy as np
import re
import torch
from torch_geometric.data import Dataset, Data
import os


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
    def __init__(self, input_dir, label_dir, transform=None, pre_transform=None):
        self.input_dir = input_dir
        self.label_dir = label_dir
        super(GraphDataset, self).__init__(transform, pre_transform)

    @property
    def raw_file_names(self):
        return os.listdir(self.input_dir)

    @property
    def processed_file_names(self):
        return os.listdir(self.input_dir)

    def len(self):
        return len(self.raw_file_names)

    def get(self, idx):
        graph_file = self.raw_file_names[idx]
        graph_path = os.path.join(self.input_dir, graph_file)

        graph_data = torch.load(graph_path)

        input_data = torch.load(os.path.join(self.input_dir, graph_file))

        label_data = torch.load(os.path.join(self.label_dir, graph_file))

        data = Data(x=input_data,
                    edge_index=graph_data.edge_index,
                    y=label_data)

        return data

