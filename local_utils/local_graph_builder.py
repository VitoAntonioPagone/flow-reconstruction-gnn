import numpy as np
import os
import torch
from torch_geometric.data import Data
from scipy.spatial import cKDTree
import sys
from torch_geometric.utils import to_undirected

# Define the missing percentage value
MISSING_PERCENTAGE = 90

def load_npz_data(file_path):
    print(f"Loading data from: {file_path}")
    with np.load(file_path) as data:
        x = data['x']
        y = data['y']
        x_velocity = data['x_velocity']
        y_velocity = data['y_velocity']
        z_velocity = data['z_velocity']
        
    features = np.column_stack((x_velocity, y_velocity, z_velocity))
    
    # Add binary indicator feature for missing data
    missing_indicator = np.linalg.norm(features, axis=1) == 0
    features = np.column_stack((features, missing_indicator))
    
    coordinates = np.column_stack((x, y))
    
    # Add coordinates to the features tensor
    features = np.column_stack((features, coordinates))

    return torch.tensor(features, dtype=torch.float), torch.tensor(coordinates, dtype=torch.float)

def create_and_save_graph(features, coordinates, num_neighbours, folder, file_base, is_input):
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
    suffix = f"_input_{MISSING_PERCENTAGE}.pt" if is_input else f"_label_{MISSING_PERCENTAGE}.pt"
    file_path = os.path.join(folder, f"{file_base}{suffix}")
    torch.save(graph, file_path)

def create_graphs(data_folder, num_neighbours, save_folder, is_input):
    for i, file in enumerate(os.listdir(data_folder)):
        if file.endswith(".npz"):
            file_path = os.path.join(data_folder, file)
            file_base = os.path.splitext(file)[0]  # Remove file extension to get the base file name
            print(f"Analyzing file: {file_path}")
            features, coordinates = load_npz_data(file_path)
            create_and_save_graph(features, coordinates, num_neighbours, save_folder, file_base, is_input)

# Configuring path and number of neighbours
num_neighbours = 8

test_data   = "../dataset_graph/original_data/npz_data/test"
test_inputs = f"../dataset_graph/original_data/npz_data/test_inputs_{MISSING_PERCENTAGE}"
os.makedirs(test_inputs, exist_ok=True)

save_graphs_folder = "../dataset_graph/training"
# Flush stdout after each print statement
sys.stdout.flush()

create_graphs(test_data, num_neighbours, os.path.join(save_graphs_folder, f"test_label_graphs_{MISSING_PERCENTAGE}"), is_input=False)
print("Test graphs created.")
sys.stdout.flush()

create_graphs(test_inputs, num_neighbours, os.path.join(save_graphs_folder, f"test_input_graphs_{MISSING_PERCENTAGE}"), is_input=True)
print("Test input graphs created.")
sys.stdout.flush()
