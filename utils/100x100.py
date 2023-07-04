import numpy as np
import os
import random
import torch
from torch_geometric.data import Data
from scipy.spatial import cKDTree
from torch_geometric.utils import to_undirected
from scipy.interpolate import griddata

# Constants
NUM_NEIGHBOURS = 8
PERCENTAGE = 0.5  
GRID_SIZE = 100
SLICE_PATH = "../dataset/original_data/npz_data/test/cyc10_CAD615_Y0_Z0_X0.npz"  
SAVE_GRAPHS_FOLDER = "../dataset/original_data/onehundredfile/" 

def load_npz_data(file_path):
    print(f"Loading data from: {file_path}")
    with np.load(file_path) as data:
        x = data['x']
        y = data['y']
        x_velocity = data['x_velocity']
        y_velocity = data['y_velocity']
        z_velocity = data['z_velocity']

    # Perform interpolation on a regular grid
    grid_x, grid_y = np.mgrid[min(x):max(x):GRID_SIZE*1j, min(y):max(y):GRID_SIZE*1j]
    grid_x_velocity = griddata((x, y), x_velocity, (grid_x, grid_y), method='nearest')
    grid_y_velocity = griddata((x, y), y_velocity, (grid_x, grid_y), method='nearest')
    grid_z_velocity = griddata((x, y), z_velocity, (grid_x, grid_y), method='nearest')

    # Flatten the data for later processes
    x = grid_x.flatten()
    y = grid_y.flatten()
    x_velocity = grid_x_velocity.flatten()
    y_velocity = grid_y_velocity.flatten()
    z_velocity = grid_z_velocity.flatten()

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

def interpolate_and_process_slice(slice_path):
    features, coordinates = load_npz_data(slice_path)

    # Save the complete graph (label graph)
    create_and_save_graph(features, coordinates, NUM_NEIGHBOURS, SAVE_GRAPHS_FOLDER, is_input=False)

    # Select indices of 50% of the points randomly
    num_points = int(features.shape[0] * PERCENTAGE)
    indices = random.sample(range(features.shape[0]), num_points)

    # Set the velocity features of the selected points to zero in the input graph
    features_input = features.clone()
    features_input[indices, :3] = 0

    # Create and save the input graph
    create_and_save_graph(features_input, coordinates, NUM_NEIGHBOURS, SAVE_GRAPHS_FOLDER, is_input=True)

# Run the function for the slice
interpolate_and_process_slice(SLICE_PATH)
