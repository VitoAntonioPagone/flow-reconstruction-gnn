import torch
import numpy as np
from scipy.interpolate import griddata
import matplotlib.pyplot as plt

# Load the .pt file
graph = torch.load("../dataset_graph/training/test_input_graphs/graph_3_input.pt")

# Get node features
node_features = graph.x

# Check the shape of the loaded tensor
print(f"Loaded node features shape: {node_features.shape}")

# Assuming that the positions are the last 2 features in the feature vector
positions = node_features[:, -2:].numpy()

# Define grid size
grid_size = 512  # Increased for a smoother plot

# Get minimum and maximum position values
min_x, min_y = np.min(positions[:, 0]), np.min(positions[:, 1])
max_x, max_y = np.max(positions[:, 0]), np.max(positions[:, 1])

# Create the grid
grid_x, grid_y = np.mgrid[min_x:max_x:grid_size*1j, min_y:max_y:grid_size*1j]

fig, axs = plt.subplots(1, 3, figsize=(18, 6))  # 1 row for 3 channels (velocities)

for i in range(3):  # iterate over velocity channels
    velocities = node_features[:, i].numpy()  # Retrieve the velocity for the current channel

    # Interpolate the values onto the regular grid
    grid_velocities = griddata(positions, velocities, (grid_x, grid_y), method='nearest')

    # Plotting the grid
    im = axs[i].imshow(grid_velocities.T, extent=(min_x, max_x, min_y, max_y), origin='lower', cmap='jet')
    axs[i].set_xlim(min_x, max_x)
    axs[i].set_ylim(min_y, max_y)
    axs[i].set_title(f'Interpolated Velocity Channel {i+1}')
    fig.colorbar(im, ax=axs[i], orientation='vertical')

plt.tight_layout()
plt.show()
