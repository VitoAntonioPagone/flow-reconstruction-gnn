import numpy as np
import os
import torch
from torch_geometric.data import Data
from scipy.spatial import cKDTree

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
    return torch.tensor(features, dtype=torch.float), torch.tensor(coordinates, dtype=torch.float)

def create_and_save_graph(features, coordinates, num_neighbours, folder, i):
    print("Creating graph...")
    tree = cKDTree(coordinates.numpy())
    distances, indices = tree.query(coordinates.numpy(), k=num_neighbours+1)

    edge_index = []
    edge_attr = []
    for v in range(len(indices)):
        for j, neighbor in enumerate(indices[v]):
            if neighbor != v:  # remove self-connections
                edge_index.append([v, neighbor])
                edge_attr.append(distances[v][j])
                
    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
    edge_attr = torch.tensor(edge_attr, dtype=torch.float).view(-1, 1)  # reshaping to [num_edges, num_edge_features]

    graph = Data(x=features, edge_index=edge_index, edge_attr=edge_attr)

    # Save the graph
    os.makedirs(folder, exist_ok=True)
    file_path = os.path.join(folder, f"graph_{i}.pt")
    torch.save(graph, file_path)

def create_graphs(data_folder, num_neighbours, save_folder):
    for i, file in enumerate(os.listdir(data_folder)):
        if file.endswith(".npz"):
            file_path = os.path.join(data_folder, file)
            print(f"Analyzing file: {file_path}")
            features, coordinates = load_npz_data(file_path)
            create_and_save_graph(features, coordinates, num_neighbours, save_folder, i)

# Configuring path and number of neighbours
num_neighbours = 6

train_data = "../dataset_graph/original_data/npz_data/train"
test_data = "../dataset_graph/original_data/npz_data/test"
validation_data = "../dataset_graph/original_data/npz_data/validation"
train_inputs = "../dataset_graph/original_data/npz_data/train_inputs"
test_inputs = "../dataset_graph/original_data/npz_data/test_inputs"
validation_inputs = "../dataset_graph/original_data/npz_data/validation_inputs"

save_graphs_folder = "../dataset_graph/training"
create_graphs(train_data, num_neighbours, os.path.join(save_graphs_folder, "train_graphs"))
print("Train graphs created.")
create_graphs(test_data, num_neighbours, os.path.join(save_graphs_folder, "test_graphs"))
print("Test graphs created.")
create_graphs(validation_data, num_neighbours, os.path.join(save_graphs_folder, "validation_graphs"))
print("Validation graphs created.")
create_graphs(train_inputs, num_neighbours, os.path.join(save_graphs_folder, "train_input_graphs"))
print("Train input graphs created.")
create_graphs(test_inputs, num_neighbours, os.path.join(save_graphs_folder, "test_input_graphs"))
print("Test input graphs created.")
create_graphs(validation_inputs, num_neighbours, os.path.join(save_graphs_folder, "validation_input_graphs"))
print("Validation input graphs created.")
