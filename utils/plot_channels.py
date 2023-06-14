import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import griddata

# Replace the filename with the path to your NPZ file
INPUT = '../dataset_graph/original_data/npz_data/train_inputs/cyc11_CAD605_Y0_Z0_X0.npz'
LABEL = '../dataset_graph/original_data/npz_data/train_inputs/cyc11_CAD605_Y0_Z0_X0.npz'

# Prepare the regular grid
grid_x, grid_y = np.mgrid[0:1:256j, 0:1:256j]

def plot_npz_channels(input_file, label_file):
    input_data = np.load(input_file)
    label_data = np.load(label_file)

    x = input_data['x']
    y = input_data['y']
    positions = np.column_stack((x, y))

    velocity_names = ['x_velocity', 'y_velocity', 'z_velocity']

    for i, name in enumerate(velocity_names):
        velocity = input_data[name]

        # Interpolate the values onto the regular grid
        grid_velocity = griddata(positions, velocity, (grid_x, grid_y), method='nearest')

        fig, axs = plt.subplots(1, 2, figsize=(18, 6))  # 1 row, 2 columns

        # Input data
        im1 = axs[0].imshow(grid_velocity, extent=(0, 1, 0, 1), origin='lower', cmap='jet')
        axs[0].set_title(f'Input: {name}')
        fig.colorbar(im1, ax=axs[0])

        # Label data
        im2 = axs[1].imshow(label_data[:, :, i], extent=(0, 1, 0, 1), origin='lower', cmap='jet')
        axs[1].set_title(f'Label: Channel {i+1}')
        fig.colorbar(im2, ax=axs[1])

        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    plot_npz_channels(INPUT, LABEL)
