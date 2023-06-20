import torch
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import griddata

# Load the .pt file
graph = torch.load("../dataset_graph/training/test_label_graphs/cyc10_CAD615_Y0_Z0_X0_label.pt")

# Get node features
node_features = graph.x

# Check the shape of the loaded tensor
print(f"Loaded node features shape: {node_features.shape}")

# Assuming the first 3 features are velocities
velocities = node_features[:, :3].numpy()

# Calculate number of zero-velocity nodes and their percentage
zero_velocity_nodes = np.all(velocities == 0, axis=1)
percentage_zero_velocity_nodes = np.mean(zero_velocity_nodes) * 100

# Print the results
print(f"Percentage of nodes with zero velocity: {percentage_zero_velocity_nodes:.2f}%")


# Assuming that the positions are the last 2 features in the feature vector
positions = node_features[:, -2:].numpy()

# Define grid size
grid_size = 256  # Increased for a smoother plot

# Get minimum and maximum position values
min_x, min_y = np.min(positions[:, 0]), np.min(positions[:, 1])
max_x, max_y = np.max(positions[:, 0]), np.max(positions[:, 1])

# Create the grid
grid_x, grid_y = np.mgrid[min_x:max_x:grid_size*1j, min_y:max_y:grid_size*1j]

fig, axs = plt.subplots(1, 3, figsize=(18, 6))  # 1 row for 3 channels (velocities)

for i in range(3):  # iterate over velocity channels
    velocities = node_features[:, i].numpy()  # Retrieve the velocity for the current channel

    # Interpolate the values onto the regular grid
    grid_velocities = griddata(positions, velocities, (grid_x, grid_y), method='linear')

    # Plotting the grid
    im = axs[i].imshow(grid_velocities.T, extent=(min_x, max_x, min_y, max_y), origin='lower', cmap='jet')
    axs[i].set_xlim(min_x, max_x)
    axs[i].set_ylim(min_y, max_y)
    axs[i].set_title(f'Interpolated Velocity Channel {i+1}')
    fig.colorbar(im, ax=axs[i], orientation='vertical')

plt.tight_layout()
plt.show()
