import numpy as np
from scipy.interpolate import griddata
import matplotlib.pyplot as plt

# Load the .npz file
data = np.load("../dataset_graph/original_data/npz_data/train_inputs/cyc11_CAD605_Y0_Z0_X0.npz")

# Get node features
x = data['x']
y = data['y']
x_velocity = data['x_velocity']
y_velocity = data['y_velocity']
z_velocity = data['z_velocity']

positions = np.column_stack((x, y))
velocities = np.column_stack((x_velocity, y_velocity, z_velocity))

# Define grid size
grid_size = 512  # Increased for a smoother plot

# Get minimum and maximum position values
min_x, min_y = np.min(positions[:, 0]), np.min(positions[:, 1])
max_x, max_y = np.max(positions[:, 0]), np.max(positions[:, 1])

# Create the grid
grid_x, grid_y = np.mgrid[min_x:max_x:grid_size*1j, min_y:max_y:grid_size*1j]

fig, axs = plt.subplots(1, 3, figsize=(18, 6))  # 1 row for 3 channels (x, y, z velocities)

for i in range(3):  # iterate over velocity channels
    channel_velocities = velocities[:, i]

    # Interpolate the values onto the regular grid
    grid_velocities = griddata(positions, channel_velocities, (grid_x, grid_y), method='nearest')

    # Plotting the grid
    im = axs[i].imshow(grid_velocities.T, extent=(min_x, max_x, min_y, max_y), origin='lower', cmap='jet')
    axs[i].set_xlim(min_x, max_x)
    axs[i].set_ylim(min_y, max_y)
    axs[i].set_title(f'Interpolated Velocity Channel {i+1}')
    fig.colorbar(im, ax=axs[i], orientation='vertical')

plt.tight_layout()
plt.show()
