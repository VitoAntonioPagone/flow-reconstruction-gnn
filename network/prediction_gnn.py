import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
from torch.utils.data import Dataset as TorchDataset
from models import AGNN_90, GAT_95, GraphSAGE_95
MISSING_PERCENTAGE = 95
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CHECKPOINT_PATH = '../trained_models/GAT_95_epochs_50_lr_0.0001_batch_1.pth.tar' 


class CustomDataset(TorchDataset):
    def __init__(self, input_files, label_files):
        self.input_files = input_files
        self.label_files = label_files

    def __len__(self):
        return len(self.input_files)

    def __getitem__(self, idx):
        input_data = torch.load(self.input_files[idx])
        label_data = torch.load(self.label_files[idx])
        input_data.y = label_data.x
        input_data.x_complete = label_data.x
        return input_data


def load_checkpoint(model, checkpoint_path):
    checkpoint = torch.load(checkpoint_path, map_location=DEVICE)
    state_dict = checkpoint['state_dict']
    model.load_state_dict(state_dict)


def rmse(pred, target):
    """Computes root mean squared error"""
    return torch.sqrt(torch.mean((pred - target) ** 2))


def mae(pred, target):
    """Computes mean absolute error"""
    return torch.mean(torch.abs(pred - target))


def run_GCN(input_file, label_file):
    # Load dataset
    test_dataset = CustomDataset([input_file], [label_file])

    # Select a single test graph
    single_graph = test_dataset[0]

    # Assuming that the positions are the last 2 features in the feature vector
    positions = single_graph.x[:, -2:].numpy()

    # Print min and max of the positions
    print(f'Min x position: {np.min(positions[:, 0])}')
    print(f'Max x position: {np.max(positions[:, 0])}')
    print(f'Min y position: {np.min(positions[:, 1])}')
    print(f'Max y position: {np.max(positions[:, 1])}')

    ######## MODEL ########
    model = GraphSAGE_95()
    ######## MODEL ########
    
    model.to(DEVICE)

    # Load trained weights
    load_checkpoint(model, CHECKPOINT_PATH)

    model.eval()

    # Move data to the correct device
    single_graph.x = single_graph.x.to(DEVICE)
    single_graph.edge_index = single_graph.edge_index.to(DEVICE)
    single_graph.y = single_graph.y.to(DEVICE)

    # Perform prediction
    with torch.no_grad():
        out = model(single_graph)

    # Check if output is entirely zero
    print("Output zero check:", torch.all(out==0).item())

    # Define grid size
    grid_size = 256  # Increased for a smoother plot

    # Get minimum and maximum position values
    min_x, min_y = np.min(positions[:, 0]), np.min(positions[:, 1])
    max_x, max_y = np.max(positions[:, 0]), np.max(positions[:, 1])

    # Create the grid
    grid_x, grid_y = np.mgrid[min_x:max_x:grid_size*1j, min_y:max_y:grid_size*1j]

    def rmse(pred, target):
        """Computes root mean squared error"""
        return torch.sqrt(torch.mean((pred - target) ** 2))

    fig, axs = plt.subplots(4, 3, figsize=(10, 10))  # Changed the subplot configuration
    fig.subplots_adjust(hspace=0.5, wspace=0.5) 
    
    def mae(pred, target):
        """Computes mean absolute error"""
        return torch.mean(torch.abs(pred - target))

    # Variables to keep track of min and max difference across all channels
    diff_min = np.inf
    diff_max = -np.inf

    channels = ['x-velocity', 'y-velocity', 'z-velocity']

    for i in range(3):
        input_values = single_graph.x.cpu()[:, i].numpy() * 7.035423
        output_values = out.cpu()[:, i].numpy() * 7.035423
        target_values = single_graph.y.cpu()[:, i].numpy() * 7.035423
        grid_input_values  = griddata(positions, input_values, (grid_x, grid_y), method='nearest')
        grid_output_values = griddata(positions, output_values, (grid_x, grid_y), method='nearest')
        grid_target_values = griddata(positions, target_values, (grid_x, grid_y), method='nearest')
        vmin, vmax = target_values.min(), target_values.max()
        pixel_wise_rmse = rmse(torch.tensor(grid_output_values), torch.tensor(grid_target_values))
        print(f"Pixel-wise RMSE for {channels[i]}: {pixel_wise_rmse}")
        pixel_wise_mae = mae(torch.tensor(grid_output_values), torch.tensor(grid_target_values))
        print(f"Pixel-wise MAE for {channels[i]}: {pixel_wise_mae}")

        im = axs[0, i].imshow(grid_input_values.T[::-1], extent=(min_x, max_x, min_y, max_y), origin='lower', cmap='jet', vmin=vmin, vmax=vmax)
        axs[0, i].set_title(f'Input {channels[i]}')
        axs[0, i].set_xticks([])
        axs[0, i].set_yticks([])
        cbar1 = fig.colorbar(im, ax=axs[0, i])
        cbar1.ax.tick_params(labelsize=8)

        im = axs[1, i].imshow(grid_output_values.T[::-1], extent=(min_x, max_x, min_y, max_y), origin='lower', cmap='jet', vmin=vmin, vmax=vmax)
        axs[1, i].set_title(f'Output {channels[i]}')
        axs[1, i].set_xticks([])
        axs[1, i].set_yticks([])
        cbar2 = fig.colorbar(im, ax=axs[1, i])
        cbar2.ax.tick_params(labelsize=8)

        im = axs[2, i].imshow(grid_target_values.T[::-1], extent=(min_x, max_x, min_y, max_y), origin='lower', cmap='jet', vmin=vmin, vmax=vmax)
        axs[2, i].set_title(f'Target {channels[i]}')
        axs[2, i].set_xticks([])
        axs[2, i].set_yticks([])
        cbar3 = fig.colorbar(im, ax=axs[2, i])
        cbar3.ax.tick_params(labelsize=8)

        diff_min = min(diff_min, np.min(grid_output_values - grid_target_values))
        diff_max = max(diff_max, np.max(grid_output_values - grid_target_values))

    for i in range(3):
        # Use the scaled data for calculating the difference
        scaled_output_values = out.cpu()[:, i].numpy() * 7.035423
        scaled_target_values = single_graph.y.cpu()[:, i].numpy() * 7.035423

        # Calculate the difference using scaled data
        diff_values = scaled_output_values - scaled_target_values

        # Then create the grid for this difference
        grid_diff_values = griddata(positions, diff_values, (grid_x, grid_y), method='nearest')

        im = axs[3, i].imshow(grid_diff_values.T[::-1], extent=(min_x, max_x, min_y, max_y), origin='lower', cmap='jet', vmin=diff_min, vmax=diff_max)
        axs[3, i].set_title(f'Difference {channels[i]}')
        axs[3, i].set_xticks([])
        axs[3, i].set_yticks([])
        cbar4 = fig.colorbar(im, ax=axs[3, i], orientation='vertical')
        cbar4.ax.tick_params(labelsize=8)

    plt.tight_layout()
    fig.savefig('../results/{}_plot.png'.format(os.path.basename(CHECKPOINT_PATH).split('.')[0]), dpi=300)
    plt.show()

if __name__ == "__main__":
    input_file = f'../dataset_graph/training/test_input_graphs_{MISSING_PERCENTAGE}/cyc10_CAD615_Y3_Z1_X0_input_95.pt'
    label_file = f'../dataset_graph/training/test_label_graphs_{MISSING_PERCENTAGE}/cyc10_CAD615_Y3_Z1_X0_label_95.pt'
    run_GCN(input_file, label_file)