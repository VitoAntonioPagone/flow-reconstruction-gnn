import numpy as np
from scipy.interpolate import griddata
import os
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree
import torch
from torch_geometric.data import Data
from torch_geometric.utils import to_undirected

# Set seed for reproducibility
np.random.seed(42)

GRID_SIZE = 100
PERCENT_TO_REMOVE = 50
MISSING_PERCENTAGE = 50
LABEL_FILE_PATH = "../dataset_graph/original_data/npz_data/test/cyc10_CAD615_Y0_Z0_X0.npz"
INTERPOLATED_OUTPUT_PATH = "../dataset_graph/original_data/onehundred/label_interpolated.npz"
ZEROED_OUTPUT_PATH = "../dataset_graph/original_data/onehundred/input_interpolated.npz"
SAVE_GRAPHS_FOLDER = "../dataset_graph/original_data/onehundred"

def load_label(file_path):
    data = np.load(file_path)
    return data

def interpolate_label(data):
    x_grid = np.linspace(min(data['x']), max(data['x']), GRID_SIZE)
    y_grid = np.linspace(min(data['y']), max(data['y']), GRID_SIZE)
    X_grid, Y_grid = np.meshgrid(x_grid, y_grid)

    grid_data = {'x': X_grid, 'y': Y_grid}
    for feature in data.keys():
        if feature not in ['x', 'y']:
            grid_data[feature] = griddata(
                np.array([data['x'], data['y']]).T, data[feature],
                (X_grid, Y_grid), method='nearest'
            )

    return grid_data

def load_npz_data(file_path):
    print(f"Loading data from: {file_path}")
    with np.load(file_path) as data:
        x = data['x'].flatten()
        y = data['y'].flatten()
        x_velocity = data['x_velocity'].flatten()
        y_velocity = data['y_velocity'].flatten()
        z_velocity = data['z_velocity'].flatten()

    features = np.column_stack((x_velocity, y_velocity, z_velocity))

    # Add binary indicator feature for missing data
    missing_indicator = np.linalg.norm(features, axis=1) == 0
    features = np.column_stack((features, missing_indicator))

    coordinates = np.column_stack((x, y))

    # Add coordinates to the features tensor
    features = np.column_stack((features, coordinates))

    return torch.tensor(features, dtype=torch.float), torch.tensor(coordinates, dtype=torch.float)

def save_label(data, output_file_path):
    directory = os.path.dirname(output_file_path)
    if not os.path.exists(directory):
        os.makedirs(directory)
    np.savez_compressed(output_file_path, **data)

def set_velocity_to_zero(data):
    percent_to_zero = int(PERCENT_TO_REMOVE / 100 * np.prod(data['x_velocity'].shape))
    indices_to_zero = np.random.choice(data['x_velocity'].size, percent_to_zero, replace=False)
    data['x_velocity'].flat[indices_to_zero] = 0
    data['y_velocity'].flat[indices_to_zero] = 0
    data['z_velocity'].flat[indices_to_zero] = 0
    return data

def plot_velocity_channels(data, title):
    fig, axs = plt.subplots(3, figsize=(8, 12))

    axs[0].imshow(data['x_velocity'], cmap='jet')
    axs[0].set_title('X Velocity')

    axs[1].imshow(data['y_velocity'], cmap='jet')
    axs[1].set_title('Y Velocity')

    axs[2].imshow(data['z_velocity'], cmap='jet')
    axs[2].set_title('Z Velocity')

    fig.suptitle(title, fontsize=16)
    plt.show()


def create_and_save_graph(features, coordinates, num_neighbours, folder, file_base, is_input):
    print("Creating graph...")
    tree = cKDTree(coordinates.numpy())
    _, indices = tree.query(coordinates.numpy(), k=num_neighbours+1)

    edge_index = []
    for v in range(len(indices)):
        for neighbor in indices[v]:
            if neighbor != v:  # remove self-connections
                edge_index.append([v, neighbor])

    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()

    # Convert to undirected graph
    edge_index = to_undirected(edge_index)

    graph = Data(x=features, edge_index=edge_index)

    # Save the graph
    os.makedirs(folder, exist_ok=True)
    suffix = f"_input_{MISSING_PERCENTAGE}.pt" if is_input else f"_label_{MISSING_PERCENTAGE}.pt"
    file_path = os.path.join(folder, f"{file_base}{suffix}")
    torch.save(graph, file_path)

def main():
    print("Starting the main function...")

    print(f"Processing NPZ file: {LABEL_FILE_PATH}")

    data = load_label(LABEL_FILE_PATH)
    data_interpolated = interpolate_label(data)

    save_label(data_interpolated, INTERPOLATED_OUTPUT_PATH)
    plot_velocity_channels(data_interpolated, 'Label Data')

    features, coordinates = load_npz_data(INTERPOLATED_OUTPUT_PATH)
    create_and_save_graph(features, coordinates, num_neighbours=8,
                          folder=os.path.join(SAVE_GRAPHS_FOLDER, f"test_label_graphs_{MISSING_PERCENTAGE}"),
                          file_base='label_interpolated', is_input=False)

    data_interpolated_zeroed = set_velocity_to_zero(data_interpolated)

    save_label(data_interpolated_zeroed, ZEROED_OUTPUT_PATH)
    plot_velocity_channels(data_interpolated_zeroed, 'Input Data')

    features_zeroed, coordinates_zeroed = load_npz_data(ZEROED_OUTPUT_PATH)
    create_and_save_graph(features_zeroed, coordinates_zeroed, num_neighbours=8,
                          folder=os.path.join(SAVE_GRAPHS_FOLDER, f"test_input_graphs_{MISSING_PERCENTAGE}"),
                          file_base='input_interpolated', is_input=True)

    print("Finished the main function.")

if __name__ == "__main__":
    main()
