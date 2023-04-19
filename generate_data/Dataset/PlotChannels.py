import numpy as np
import matplotlib.pyplot as plt

# Replace the filename with the path to your NPZ file
label = 'Dataset/train/train-labels-50/interpolated_cyc11_CAD605_Y0_Z0_X0_label.npy'
train = 'Dataset/train/train-50/interpolated_cyc11_CAD605_Y0_Z0_X0_train.npy'

def plot_npy_channels(file1, file2):
    data1 = np.load(file1)
    data2 = np.load(file2)

    n_channels = data1.shape[-1]

    for channel in range(n_channels):
        fig, axes = plt.subplots(1, 2, figsize=(15, 5))

        # Plot data from file1
        im1 = axes[0].imshow(data1[:, :, channel], cmap='jet', aspect='auto')
        axes[0].set_title(f'File1: Channel {channel}')
        cbar1 = fig.colorbar(im1, ax=axes[0])
        cbar1.set_label('Values')

        # Plot data from file2
        im2 = axes[1].imshow(data2[:, :, channel], cmap='jet', aspect='auto')
        axes[1].set_title(f'File2: Channel {channel}')
        cbar2 = fig.colorbar(im2, ax=axes[1])
        cbar2.set_label('Values')

        plt.show()

# Replace 'label' and 'train' with the paths to your .npy files
plot_npy_channels(label, train)
