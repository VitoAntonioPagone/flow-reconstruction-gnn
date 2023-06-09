import sys
import torch
import numpy as np
import pandas as pd
import os
import glob
import matplotlib.pyplot as plt
from datasets import FlowDataset
from models import ConvAutoEncoder_simplified_50
from utils import load_checkpoint

def calculate_rmse(pred, target):
    """Calculate RMSE"""
    return torch.sqrt(((pred - target) ** 2).mean())

def run_autoencoder():
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    CHECKPOINT_FILE = '../trained_models/ConvAutoEncoder_simplified_epochs_1000_autoencoder_checkpoint_alpha_0.1_beta_0.1_lr_0.0001_batch_128.pth.tar'
    TEST_INPUTS_DIR = '../dataset/train_data_50/test_inputs_50/'
    TEST_LABELS_DIR = '../dataset/train_data_50/test_labels_50/'
    TEST_FILE_NUM = 8


    def reconstruct_flow(model, input_tensor, missing_mask_tensor):
        with torch.no_grad():
            # Concatenate the input tensor and the single-channel missing mask tensor
            input_with_mask = torch.cat((input_tensor, missing_mask_tensor), dim=1)
            output_tensor = model(input_with_mask)
            reconstructed_flow = output_tensor * missing_mask_tensor + input_tensor * (1 - missing_mask_tensor)
        return reconstructed_flow


    def predict():
        # Get test data set
        test_dataset = FlowDataset(TEST_INPUTS_DIR, TEST_LABELS_DIR)
        # Get a sample input tensor, ground truth label tensor, and its single-channel missing mask tensor
        test_input_tensor, test_label_tensor, test_missing_mask_tensor = test_dataset[TEST_FILE_NUM]
        test_input_tensor = test_input_tensor.to(DEVICE)
        test_label_tensor = test_label_tensor.to(DEVICE)
        test_missing_mask_tensor = test_missing_mask_tensor.to(DEVICE)
        # Add the batch dimension
        test_input_tensor = test_input_tensor.unsqueeze(0)
        test_label_tensor = test_label_tensor.unsqueeze(0)
        test_missing_mask_tensor = test_missing_mask_tensor.unsqueeze(0)
        # Load pre-trained model
        model = ConvAutoEncoder_simplified_50().to(DEVICE)
        load_checkpoint(torch.load(CHECKPOINT_FILE, map_location=torch.device(DEVICE)), model)
        reconstructed_flow_tensor = reconstruct_flow(model, test_input_tensor, test_missing_mask_tensor)
        reconstructed_flow_tensor = reconstructed_flow_tensor.cpu()  # Move the tensor back to CPU for visualization
        print("Reconstructed Flow Tensor Dimension:", reconstructed_flow_tensor.size())
        fig, axes = plt.subplots(4, 3, figsize=(12, 12))  # Changed the subplot configuration
        fig.subplots_adjust(hspace=0.5, wspace=0.5) 

        for i in range(3):
            rmse = calculate_rmse(reconstructed_flow_tensor[0, i], test_label_tensor[0, i])
            print(f"Channel {i + 1} RMSE: {rmse.item()}")
            # Find min and max values of the ground truth tensor for the current channel
            vmin = test_label_tensor[0, i].min()
            vmax = test_label_tensor[0, i].max()

            # Plot the input tensor
            im1 = axes[0, i].imshow(test_input_tensor[0, i], cmap='jet', aspect='auto', vmin=vmin, vmax=vmax)
            axes[0, i].set_title(f"Input Tensor - Channel {i + 1}", fontsize=10)
            cbar1 = fig.colorbar(im1, ax=axes[0, i], shrink=0.6)
            cbar1.ax.tick_params(labelsize=8)

            # Plot the ground truth label tensor
            im2 = axes[1, i].imshow(test_label_tensor[0, i], cmap='jet', aspect='auto', vmin=vmin, vmax=vmax)
            axes[1, i].set_title(f"Ground Truth Label Tensor - Channel {i + 1}", fontsize=10)
            cbar2 = fig.colorbar(im2, ax=axes[1, i], shrink=0.6)
            cbar2.ax.tick_params(labelsize=8)

            # Plot the reconstructed flow tensor
            im3 = axes[2, i].imshow(reconstructed_flow_tensor[0, i], cmap='jet', aspect='auto', vmin=vmin, vmax=vmax)
            axes[2, i].set_title(f"Reconstructed Flow Tensor - Channel {i + 1}", fontsize=10)
            cbar3 = fig.colorbar(im3, ax=axes[2, i], shrink=0.6)
            cbar3.ax.tick_params(labelsize=8)

            difference_tensor = test_label_tensor[0, i] - reconstructed_flow_tensor[0, i]
            im4 = axes[3, i].imshow(difference_tensor, cmap='jet', aspect='auto')
            axes[3, i].set_title(f"Difference Tensor - Channel {i + 1}", fontsize=10)
            cbar4 = fig.colorbar(im4, ax=axes[3, i], shrink=0.6)
            cbar4.ax.tick_params(labelsize=8)

        plt.tight_layout(pad=1)  
        plt.show()

    predict()


if __name__ == "__main__":
    run_autoencoder()
