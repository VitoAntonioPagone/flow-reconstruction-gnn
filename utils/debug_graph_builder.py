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

def create_graph(features, coordinates, num_neighbours):
    print("Creating graph...")
    tree = cKDTree(coordinates.numpy())
    distances, indices = tree.query(coordinates.numpy(), k=num_neighbours+1)  # k+1 because the point itself is always returned

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
    return graph

def create_graphs(data_folder, num_neighbours):
    file = os.listdir(data_folder)[0]  # Get the first file
    if file.endswith(".npz"):
        file_path = os.path.join(data_folder, file)
        print(f"Analyzing file: {file_path}")
        features, coordinates = load_npz_data(file_path)
        graph = create_graph(features, coordinates, num_neighbours)
        return [graph]  # Return a list with a single graph
    return []

# Configuring path and number of neighbours
num_neighbours = 4


def save_graphs(graphs, folder, is_input):
    os.makedirs(folder, exist_ok=True)
    suffix = "_input.pt" if is_input else "_label.pt"
    for i, graph in enumerate(graphs):
        file_path = os.path.join(folder, f"graph_{i}{suffix}")
        torch.save(graph, file_path)


# Configuring paths
train_data = "../dataset_graph/original_data/npz_data/train"
train_inputs = "../dataset_graph/original_data/npz_data/train_inputs"
save_folder = "../dataset_graph/train_graphs"

# Creating and saving the full train graphs
train_graphs = create_graphs(train_data, num_neighbours)
save_graphs(train_graphs, save_folder, is_input=False)

# Creating and saving the input train graphs
train_input_graphs = create_graphs(train_inputs, num_neighbours)
save_graphs(train_input_graphs, save_folder, is_input=True)


