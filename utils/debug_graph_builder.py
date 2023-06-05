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
    coordinates = np.column_stack((x, y))
    return torch.tensor(features, dtype=torch.float), torch.tensor(coordinates, dtype=torch.float)

def create_graph(features, coordinates, distance_threshold):
    print("Creating graph...")
    tree = cKDTree(coordinates.numpy())
    adjacency_matrix = tree.query_ball_tree(tree, distance_threshold)

    edge_index = []
    for v in range(len(adjacency_matrix)):
        for neighbor in adjacency_matrix[v]:
            if neighbor != v:  # remove self-connections
                edge_index.append([v, neighbor])
    
    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()

    graph = Data(x=features, edge_index=edge_index)
    return graph

def create_graphs(data_folder, distance_threshold):
    file = os.listdir(data_folder)[0]  # Get the first file
    if file.endswith(".npz"):
        file_path = os.path.join(data_folder, file)
        print(f"Analyzing file: {file_path}")
        features, coordinates = load_npz_data(file_path)
        graph = create_graph(features, coordinates, distance_threshold)
        return [graph]  # Return a list with a single graph
    return []

def save_graphs(graphs, folder):
    os.makedirs(folder, exist_ok=True)
    for i, graph in enumerate(graphs):
        file_path = os.path.join(folder, f"graph_{i}.pt")
        torch.save(graph, file_path)

# Configuring path and threshold
data_folder = "../dataset_graph/original_data/npz_data/train"  # adjust this to your specific directory
distance_threshold = 0.005  # Set an appropriate distance threshold
save_folder = "../dataset_graph/train_graphs"  # adjust this to your specific directory

# Creating and saving the graph
graphs = create_graphs(data_folder, distance_threshold)
save_graphs(graphs, save_folder)
