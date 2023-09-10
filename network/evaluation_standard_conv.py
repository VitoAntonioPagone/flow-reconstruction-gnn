import sys
import torch
import numpy as np
import pandas as pd
import os
import glob
import matplotlib.pyplot as plt
from datasets import FlowDataset
from models import ConvNet_95, ConvNet_90, UNet, ConvNet_98
from utils import load_checkpoint
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset as TorchDataset

MISSING_PERCENTAGE = 98


class FlowDataset(TorchDataset):
    def __init__(self, input_files, label_files, add_mask=False):
        self.input_files = input_files
        self.label_files = label_files
        self.add_mask = add_mask

    def __len__(self):
        return len(self.input_files)

    def __getitem__(self, index):
        input_path = self.input_files[index]
        label_path = self.label_files[index]

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

def calculate_rmse(pred, target):
    """Calculate RMSE"""
    return torch.sqrt(((pred - target) ** 2).mean())

def calculate_mae(pred, target):
    """Calculate MAE"""
    return (torch.abs(pred - target)).mean()

def run_autoencoder(input_file, label_file):
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    CHECKPOINT_FILE = '../trained_models/ConvNet_98_epochs_500__alpha_0.1_beta_0.1_lr_0.0001_batch_64.pth.tar'
    TEST_INPUT_FILE = input_file
    TEST_LABEL_FILE = label_file

    def reconstruct_flow(model, input_tensor, missing_mask_tensor):
        with torch.no_grad():
            input_with_mask = torch.cat((input_tensor, missing_mask_tensor), dim=1)
            output_tensor = model(input_with_mask)
            reconstructed_flow = output_tensor * missing_mask_tensor + input_tensor * (1 - missing_mask_tensor)
        return reconstructed_flow

    def predict():
        test_dataset = FlowDataset([input_file], [label_file])
        test_input_tensor, test_label_tensor, test_missing_mask_tensor = test_dataset[0]
        test_input_tensor = test_input_tensor.to(DEVICE)
        test_label_tensor = test_label_tensor.to(DEVICE)
        test_missing_mask_tensor = test_missing_mask_tensor.to(DEVICE)
        test_input_tensor = test_input_tensor.unsqueeze(0)
        test_label_tensor = test_label_tensor.unsqueeze(0)
        test_missing_mask_tensor = test_missing_mask_tensor.unsqueeze(0)

        model = ConvNet_98().to(DEVICE)
        load_checkpoint(torch.load(CHECKPOINT_FILE, map_location=torch.device(DEVICE)), model)
        reconstructed_flow_tensor = reconstruct_flow(model, test_input_tensor, test_missing_mask_tensor)

        rmses = []
        maes = []
        
        for i in range(3):
            reconstructed_flow_tensor_denorm = reconstructed_flow_tensor[0, i] * 7.035423
            test_label_tensor_denorm = test_label_tensor[0, i] * 7.035423
            rmse = calculate_rmse(reconstructed_flow_tensor_denorm, test_label_tensor_denorm)
            mae = calculate_mae(reconstructed_flow_tensor_denorm, test_label_tensor_denorm)
            rmses.append(rmse.item())
            maes.append(mae.item())

        return rmses, maes

    # Call the function
    return predict()

if __name__ == "__main__":
    input_folder = f'../dataset/train_data_{MISSING_PERCENTAGE}/test_inputs_{MISSING_PERCENTAGE}/'
    label_folder = f'../dataset/train_data_{MISSING_PERCENTAGE}/test_labels_{MISSING_PERCENTAGE}/'
    all_rmses = []
    all_maes = []
    input_files = sorted(glob.glob(os.path.join(input_folder, "*.npy")))
    label_files = sorted(glob.glob(os.path.join(label_folder, "*.npy")))
    channel_names = ['x-velocity', 'y-velocity', 'z-velocity']

    for inp, lbl in zip(input_files, label_files):
        rmses, maes = run_autoencoder(inp, lbl)
        
        print(f"\nAnalyzing Files: {os.path.basename(inp)} and {os.path.basename(lbl)}")
        for i, channel in enumerate(channel_names):
            print(f"{channel} RMSE: {rmses[i]}")
            print(f"{channel} MAE: {maes[i]}")
            
        all_rmses.append(rmses)
        all_maes.append(maes)

    all_rmses = np.array(all_rmses)
    all_maes = np.array(all_maes)
    mean_rmses = all_rmses.mean(axis=0)
    mean_maes = all_maes.mean(axis=0)

    print("\nOverall Results:")
    for i, channel in enumerate(channel_names):
        print(f"{channel} Mean RMSE: {mean_rmses[i]}")
        print(f"{channel} Mean MAE: {mean_maes[i]}")