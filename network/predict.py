import sys
import torch
import numpy as np
import pandas as pd
import os
import glob
import matplotlib.pyplot as plt
from datasets import FlowDataset
from models import ConvAutoEncoder_simplified_50, ConvAutoEncoder_simplified_90
from utils import load_checkpoint
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset as TorchDataset


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
    CHECKPOINT_FILE = '../trained_models/ConvAutoEncoder_simplified_90_epochs_500_autoencoder_checkpoint_alpha_0.1_beta_0.1_lr_0.0001_batch_128.pth.tar'
    TEST_INPUT_FILE = input_file
    TEST_LABEL_FILE = label_file

    def reconstruct_flow(model, input_tensor, missing_mask_tensor):
        with torch.no_grad():
            # Concatenate the input tensor and the single-channel missing mask tensor
            input_with_mask = torch.cat((input_tensor, missing_mask_tensor), dim=1)
            output_tensor = model(input_with_mask)
            reconstructed_flow = output_tensor * missing_mask_tensor + input_tensor * (1 - missing_mask_tensor)
        return reconstructed_flow

    def predict():
        # Get test data set
        test_dataset = FlowDataset([input_file], [label_file])
        # Get a sample input tensor, ground truth label tensor, and its single-channel missing mask tensor
        test_input_tensor, test_label_tensor, test_missing_mask_tensor = test_dataset[0]
        test_input_tensor = test_input_tensor.to(DEVICE)
        test_label_tensor = test_label_tensor.to(DEVICE)
        test_missing_mask_tensor = test_missing_mask_tensor.to(DEVICE)
        # Add the batch dimension
        test_input_tensor = test_input_tensor.unsqueeze(0)
        test_label_tensor = test_label_tensor.unsqueeze(0)
        test_missing_mask_tensor = test_missing_mask_tensor.unsqueeze(0)
        # Load pre-trained model
        model = ConvAutoEncoder_simplified_90().to(DEVICE)
        load_checkpoint(torch.load(CHECKPOINT_FILE, map_location=torch.device(DEVICE)), model)
        reconstructed_flow_tensor = reconstruct_flow(model, test_input_tensor, test_missing_mask_tensor)
        reconstructed_flow_tensor = reconstructed_flow_tensor.cpu()  # Move the tensor back to CPU for visualization
        print("Reconstructed Flow Tensor Dimension:", reconstructed_flow_tensor.size())
        fig, axes = plt.subplots(4, 3, figsize=(10, 10))  # Changed the subplot configuration
        fig.subplots_adjust(hspace=0.5, wspace=0.5) 

        channel_names = ['x-velocity', 'y-velocity', 'z-velocity']
    
        for i in range(3):
            rmse = calculate_rmse(reconstructed_flow_tensor[0, i], test_label_tensor[0, i])
            mae = calculate_mae(reconstructed_flow_tensor[0, i], test_label_tensor[0, i])
            print(f"{channel_names[i]} RMSE: {rmse.item()}")
            print(f"{channel_names[i]} MAE: {mae.item()}")
            vmin = test_label_tensor[0, i].min()
            vmax = test_label_tensor[0, i].max()

            # Plot the input tensor
            im1 = axes[0, i].imshow(test_input_tensor[0, i].cpu(), cmap='jet', aspect='auto', vmin=vmin, vmax=vmax)
            axes[0, i].set_title(f"Input Tensor ({channel_names[i]})", fontsize=10)
            axes[0, i].set_xticks([])
            axes[0, i].set_yticks([])
            cbar1 = fig.colorbar(im1, ax=axes[0, i])
            cbar1.ax.tick_params(labelsize=8)

            # Plot the ground truth label tensor
            im2 = axes[2, i].imshow(test_label_tensor[0, i].cpu(), cmap='jet', aspect='auto', vmin=vmin, vmax=vmax)
            axes[2, i].set_title(f"Ground Truth ({channel_names[i]})", fontsize=10)
            axes[2, i].set_xticks([])
            axes[2, i].set_yticks([])
            cbar2 = fig.colorbar(im2, ax=axes[2, i])
            cbar2.ax.tick_params(labelsize=8)

            # Plot the reconstructed flow tensor
            im3 = axes[1, i].imshow(reconstructed_flow_tensor[0, i], cmap='jet', aspect='auto', vmin=vmin, vmax=vmax)
            axes[1, i].set_title(f"Reconstructed Flow ({channel_names[i]})", fontsize=10)
            axes[1, i].set_xticks([])
            axes[1, i].set_yticks([])
            cbar3 = fig.colorbar(im3, ax=axes[1, i])
            cbar3.ax.tick_params(labelsize=8)

            # Plot the difference tensor
            difference_tensor = test_label_tensor[0, i] - reconstructed_flow_tensor[0, i]
            im4 = axes[3, i].imshow(difference_tensor.cpu(), cmap='jet', aspect='auto')
            axes[3, i].set_title(f"Difference ({channel_names[i]})", fontsize=10)
            axes[3, i].set_xticks([])
            axes[3, i].set_yticks([])
            cbar4 = fig.colorbar(im4, ax=axes[3, i])
            cbar4.ax.tick_params(labelsize=8)

        plt.tight_layout(pad=1)  
        plt.show()

    # Call the function
    predict()

if __name__ == "__main__":
    input_file = '../dataset/train_data_90/test_inputs_90/interpolated_cyc10_CAD615_Y3_Z1_X0_input.npy'  
    label_file = '../dataset/train_data_90/test_labels_90/interpolated_cyc10_CAD615_Y3_Z1_X0_label.npy' 
    run_autoencoder(input_file, label_file)