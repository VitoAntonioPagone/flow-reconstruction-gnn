import numpy as np
import os
import torch
from torch_geometric.data import Data
from scipy.spatial import cKDTree
from torch_geometric.utils import to_undirected

# Constants
NUM_NEIGHBOURS = 8
INPUT_NPZ_PATH = "../dataset_graph/original_data/npz_data/test_inputs_50/cyc10_CAD615_Y0_Z0_X0.npz" 
LABEL_NPZ_PATH = "../dataset_graph/original_data/npz_data/test_labels_50/cyc10_CAD615_Y0_Z0_X0.npz" 
SAVE_GRAPHS_FOLDER = "../dataset/original_data/onehundredfile/" 

def load_npz_data(file_path):
    print(f"Loading data from: {file_path}")
    with np.load(file_path) as data:
        x = data['x']
        y = data['y']
        x_velocity = data['x_velocity']
        y_velocity = data['y_velocity']
        z_velocity = data['z_velocity']
    
    # Flatten the data for later processes
    x_velocity = x_velocity.flatten()
    y_velocity = y_velocity.flatten()
    z_velocity = z_velocity.flatten()
    
# Perform interpolation on a regular grid
# grid_x, grid_y = np.mgrid[min(x):max(x):GRID_SIZE*1j, min(y):max(y):GRID_SIZE*1j]
# grid_x_velocity = griddata((x, y), x_velocity, (grid_x, grid_y), method='nearest')
# grid_y_velocity = griddata((x, y), y_velocity, (grid_x, grid_y), method='nearest')
# grid_z_velocity = griddata((x, y), z_velocity, (grid_x, grid_y), method='nearest')

# Flatten the data for later processes
# x_velocity = grid_x_velocity.flatten()
# y_velocity = grid_y_velocity.flatten()
# z_velocity = grid_z_velocity.flatten()

    features = np.column_stack((x_velocity, y_velocity, z_velocity))
    
    # Add binary indicator feature for missing data
    missing_indicator = np.linalg.norm(features, axis=1) == 0
    features = np.column_stack((features, missing_indicator))

    coordinates = np.column_stack((x, y))
    
    # Add coordinates to the features tensor
    features = np.column_stack((features, coordinates))

    return torch.tensor(features, dtype=torch.float), torch.tensor(coordinates, dtype=torch.float)

def create_and_save_graph(features, coordinates, num_neighbours, folder, is_input):
    print("Creating graph...")
    tree = cKDTree(coordinates.numpy())
    distances, indices = tree.query(coordinates.numpy(), k=num_neighbours+1)

    edge_index = []
    for v in range(len(indices)):
        for j, neighbor in enumerate(indices[v]):
            if neighbor != v:  # remove self-connections
                edge_index.append([v, neighbor])

    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()

    # Convert to undirected graph
    edge_index = to_undirected(edge_index)

    graph = Data(x=features, edge_index=edge_index)  

    # Save the graph
    os.makedirs(folder, exist_ok=True)
    suffix = "_input.pt" if is_input else "_label.pt"
    file_path = os.path.join(folder, f"graph{suffix}")
    torch.save(graph, file_path)

# Load the input and label data from npz files
features_input, coordinates = load_npz_data(INPUT_NPZ_PATH)
features_label, _ = load_npz_data(LABEL_NPZ_PATH)


# Create and save the input graph
create_and_save_graph(features_input, coordinates, NUM_NEIGHBOURS, SAVE_GRAPHS_FOLDER, is_input=True)

# Create and save the label graph
create_and_save_graph(features_label, coordinates, NUM_NEIGHBOURS, SAVE_GRAPHS_FOLDER, is_input=False)
