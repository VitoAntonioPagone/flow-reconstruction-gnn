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
    graphs = []
    for file in os.listdir(data_folder):
        if file.endswith(".npz"):
            file_path = os.path.join(data_folder, file)
            
            print(f"Analyzing file: {file_path}")
            
            features, coordinates = load_npz_data(file_path)
            graph = create_graph(features, coordinates, distance_threshold)
            graphs.append(graph)
    return graphs

def save_graphs(graphs, folder):
    os.makedirs(folder, exist_ok=True)
    for i, graph in enumerate(graphs):
        file_path = os.path.join(folder, f"graph_{i}.pt")
        torch.save(graph, file_path)

def load_graphs(folder):
    graphs = []
    for file in os.listdir(folder):
        if file.endswith(".pt"):
            file_path = os.path.join(folder, file)
            graph = torch.load(file_path)
            graphs.append(graph)
    return graphs

distance_threshold = 0.005  # Set an appropriate distance threshold

train_data = "../dataset_graph/original_data/npz_data/train"
test_data = "../dataset_graph/original_data/npz_data/test"
validation_data = "../dataset_graph/original_data/npz_data/validation"
train_inputs = "../dataset_graph/original_data/npz_data/train_inputs"
test_inputs = "../dataset_graph/original_data/npz_data/test_inputs"
validation_inputs = "../dataset_graph/original_data/npz_data/validation_inputs"

train_graphs = create_graphs(train_data, distance_threshold)
print("Train graphs created.")
test_graphs = create_graphs(test_data, distance_threshold)
print("Test graphs created.")

validation_graphs = create_graphs(validation_data, distance_threshold)
print("Validation graphs created.")
train_input_graphs = create_graphs(train_inputs, distance_threshold)
print("Train input graphs created.")

test_input_graphs = create_graphs(test_inputs, distance_threshold)
print("Test input graphs created.")

validation_input_graphs = create_graphs(validation_inputs, distance_threshold)
print("Validation input graphs created.")

save_graphs(train_graphs, "../dataset_graph/train_graphs")
save_graphs(test_graphs, "../dataset_graph/test_graphs")
save_graphs(validation_graphs, "../dataset_graph/validation_graphs")

save_graphs(train_input_graphs, "/../dataset_graph/train_input_graphs")
save_graphs(test_input_graphs, "../dataset_graph/test_input_graphs")
save_graphs(validation_input_graphs, "../dataset_graph/validation_input_graphs")
