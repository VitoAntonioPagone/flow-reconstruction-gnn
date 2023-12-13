import numpy as np
import matplotlib.pyplot as plt

DATA_FILES = [
    '../results/CNN_ground_truth_label.npy',
    '../results/CNN_predictions.npy',
    '../results/GNNinterpolated_velocity_magnitude_GT.npy',
    '../results/GNNinterpolated_velocity_magnitude.npy'
]

def plot_npy_file(file_name, ax):
    print(f"Plotting NPY file: {file_name}")
    data = np.load(file_name)

    print(f"Data Shape: {data.shape}")
    print(f"Data Type: {data.dtype}")

    if len(data.shape) == 2:
        im = ax.imshow(data, cmap='jet')
        ax.set_title(file_name.split('/')[-1])
        ax.axis('off')
        return im
    else:
        print("Unexpected data dimensions. Unable to plot.")
        return None

if __name__ == "__main__":
    print("Starting main script execution...")

    fig, axs = plt.subplots(2, 2, figsize=(12, 12))  

    axs = axs.flatten()

    for i, data_file in enumerate(DATA_FILES):
        im = plot_npy_file(data_file, axs[i])
        if im:
            fig.colorbar(im, ax=axs[i], orientation='vertical', fraction=0.046, pad=0.04)

    plt.tight_layout()
    plt.show()  
    print("Finished main script execution.")
