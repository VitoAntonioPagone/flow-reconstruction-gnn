import torch
from datasets import CustomDataset
from models import GCN
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import griddata

# Hyperparameters
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Path
TEST_INPUT_DIR = '../dataset_graph/training/test_input_graphs'
TEST_TARGET_DIR = '../dataset_graph/training/test_graphs'
CHECKPOINT_PATH = '../trained_models/GCN_epochs_100_lr_0.001_batch_16.pth.tar' 

# Load dataset
test_dataset = CustomDataset(TEST_INPUT_DIR, TEST_TARGET_DIR)

# Select a single test graph
single_graph = test_dataset[0]

# Assuming that the positions are the last 2 features in the feature vector
positions = single_graph.x[:, -2:].numpy()

# Print min and max of the positions
print(f'Min x position: {np.min(positions[:, 0])}')
print(f'Max x position: {np.max(positions[:, 0])}')
print(f'Min y position: {np.min(positions[:, 1])}')
print(f'Max y position: {np.max(positions[:, 1])}')

model = GCN()
model.to(DEVICE)

def load_checkpoint(model, checkpoint_path):
    checkpoint = torch.load(checkpoint_path, map_location=DEVICE)
    state_dict = checkpoint['state_dict']
    model.load_state_dict(state_dict)

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

fig, axs = plt.subplots(4, 3, figsize=(16, 12))  # Change subplot configuration to 4x3

# Variables to keep track of min and max difference across all channels
diff_min = np.inf
diff_max = -np.inf

for i in range(3):  # iterate over channels
    input_values = single_graph.x.cpu()[:, i].numpy()
    output_values = out.cpu()[:, i].numpy()
    target_values = single_graph.y.cpu()[:, i].numpy()

    # Interpolate the values onto the regular grid
    grid_input_values  = griddata(positions, input_values, (grid_x, grid_y), method='nearest')
    grid_output_values = griddata(positions, output_values, (grid_x, grid_y), method='nearest')
    grid_target_values = griddata(positions, target_values, (grid_x, grid_y), method='nearest')

    # Calculate the color scale limits
    vmin, vmax = target_values.min(), target_values.max()

    # Compute the RMSE
    pixel_wise_rmse = rmse(torch.tensor(grid_output_values), torch.tensor(grid_target_values))
    print(f"Pixel-wise RMSE for Channel {i+1}: {pixel_wise_rmse}")

    # Input grid
    im = axs[0, i].imshow(grid_input_values.T, extent=(min_x, max_x, min_y, max_y), origin='lower', cmap='jet', vmin=vmin, vmax=vmax)
    axs[0, i].set_title(f'Input Channel {i+1}')
    fig.colorbar(im, ax=axs[0, i], orientation='vertical')

    # Output grid
    im = axs[1, i].imshow(grid_output_values.T, extent=(min_x, max_x, min_y, max_y), origin='lower', cmap='jet', vmin=vmin, vmax=vmax)
    axs[1, i].set_title(f'Output Channel {i+1}')
    fig.colorbar(im, ax=axs[1, i], orientation='vertical')

    # Target grid
    im = axs[2, i].imshow(grid_target_values.T, extent=(min_x, max_x, min_y, max_y), origin='lower', cmap='jet', vmin=vmin, vmax=vmax)
    axs[2, i].set_title(f'Target Channel {i+1}')
    fig.colorbar(im, ax=axs[2, i], orientation='vertical')

    # Update min and max difference if necessary
    diff_min = min(diff_min, np.min(grid_output_values - grid_target_values))
    diff_max = max(diff_max, np.max(grid_output_values - grid_target_values))

# Plot difference grids on the last row
for i in range(3):
    input_values = single_graph.x.cpu()[:, i].numpy()
    output_values = out.cpu()[:, i].numpy()
    target_values = single_graph.y.cpu()[:, i].numpy()

    # Interpolate the values onto the regular grid
    grid_input_values  = griddata(positions, input_values, (grid_x, grid_y),  method='nearest')
    grid_output_values = griddata(positions, output_values, (grid_x, grid_y), method='nearest')
    grid_target_values = griddata(positions, target_values, (grid_x, grid_y), method='nearest')

    # Difference grid
    im = axs[3, i].imshow((grid_output_values - grid_target_values).T, extent=(min_x, max_x, min_y, max_y), origin='lower', cmap='jet', vmin=diff_min, vmax=diff_max)
    axs[3, i].set_title(f'Difference Channel {i+1}')
    fig.colorbar(im, ax=axs[3, i], orientation='vertical')

plt.tight_layout()
plt.show()

