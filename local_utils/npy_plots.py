import numpy as np
import matplotlib.pyplot as plt

DATA_FILE = '../dataset/train_data_hole_128x128/flow_reconstruction/PIV_data/labels_npz/PIV_cyc_10_CAD_610.npz' 

def plot_npy_file(file_name):
    print(f"Plotting NPY file: {file_name}")
    data = np.load(file_name)

    num_channels = data.shape[2]
    channels = ['x_velocity', 'y_velocity', 'z_velocity']  # Replace with actual channel names if different

    fig, axs = plt.subplots(num_channels, 1, figsize=(10, 10))

    for i in range(num_channels):
        channel_data = data[:,:,i]
        num_zeros = np.count_nonzero(channel_data == 0)
        total_elements = channel_data.size
        percent_zeros = num_zeros / total_elements * 100
        print(f"Channel {channels[i]} has {percent_zeros:.2f}% zero features.")

        im = axs[i].imshow(channel_data, cmap='jet')
        axs[i].set_title(f'{channels[i]}')
        axs[i].axis('off')
        fig.colorbar(im, ax=axs[i], orientation='vertical', fraction=0.046, pad=0.04)

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    print("Starting main script execution...")
    plot_npy_file(DATA_FILE)
    print("Finished main script execution.")
