import torch
from datasets import CustomDataset
from models import GCN
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import griddata

# Hyperparameters
FEAT_DIM = 6
HIDDEN_DIM = 64
OUTPUT_DIM = 6
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Path
TEST_INPUT_DIR = '../dataset_graph/training/test_input_graphs'
TEST_TARGET_DIR = '../dataset_graph/training/test_graphs'
CHECKPOINT_PATH = '../trained_models/GCN_epochs_10_lr_0.01_batch_32.pth.tar' 

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

model = GCN(feat_dim=FEAT_DIM, hidden_dim=HIDDEN_DIM, output_dim=OUTPUT_DIM)
model.to(DEVICE)

def load_checkpoint(model, checkpoint_path):
    checkpoint = torch.load(checkpoint_path, map_location=DEVICE)
    state_dict = checkpoint['state_dict']
    model_dict = model.state_dict()

    # Filter out unnecessary keys
    state_dict = {k: v for k, v in state_dict.items() if k in model_dict}

    # Overwrite entries in the existing state dict
    model_dict.update(state_dict)

    # Load the new state dict
    model.load_state_dict(model_dict)

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
grid_size = 512  # Increased for a smoother plot

# Get minimum and maximum position values
min_x, min_y = np.min(positions[:, 0]), np.min(positions[:, 1])
max_x, max_y = np.max(positions[:, 0]), np.max(positions[:, 1])

# Create the grid
grid_x, grid_y = np.mgrid[min_x:max_x:grid_size*1j, min_y:max_y:grid_size*1j]

fig, axs = plt.subplots(3, 3, figsize=(18, 18))  # 3 rows for 3 channels, 3 columns for input/output/target

# Assuming that the positions are the last 2 features in the feature vector
positions = single_graph.x.cpu()[:, -2:].numpy()

for i in range(3):  # iterate over channels
    input_values = single_graph.x.cpu()[:, i].numpy()
    output_values = out.cpu()[:, i].numpy()
    target_values = single_graph.y.cpu()[:, i].numpy()

    # Interpolate the values onto the regular grid
    grid_input_values = griddata(positions, input_values, (grid_x, grid_y), method='nearest')
    grid_output_values = griddata(positions, output_values, (grid_x, grid_y), method='nearest')
    grid_target_values = griddata(positions, target_values, (grid_x, grid_y), method='nearest')

    # Calculate the color scale limits
    vmin, vmax = target_values.min(), target_values.max()

    # Input grid
    im = axs[i, 0].imshow(grid_input_values.T, extent=(min_x, max_x, min_y, max_y), origin='lower', cmap='jet', vmin=vmin, vmax=vmax)
    axs[i, 0].set_title(f'Input Channel {i+1}')
    fig.colorbar(im, ax=axs[i, 0], orientation='vertical')

    # Output grid
    im = axs[i, 1].imshow(grid_output_values.T, extent=(min_x, max_x, min_y, max_y), origin='lower', cmap='jet', vmin=vmin, vmax=vmax)
    axs[i, 1].set_title(f'Output Channel {i+1}')
    fig.colorbar(im, ax=axs[i, 1], orientation='vertical')

    # Target grid
    im = axs[i, 2].imshow(grid_target_values.T, extent=(min_x, max_x, min_y, max_y), origin='lower', cmap='jet', vmin=vmin, vmax=vmax)
    axs[i, 2].set_title(f'Target Channel {i+1}')
    fig.colorbar(im, ax=axs[i, 2], orientation='vertical')

plt.tight_layout()
plt.show()
