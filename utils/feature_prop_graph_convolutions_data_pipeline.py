import os
import glob
import shutil
import numpy as np
import vtk
import random
from torch import Tensor
import torch
import torch_geometric
from torch_geometric.data import Data
from torch_geometric.utils import to_undirected
from scipy.spatial import cKDTree
from natsort import natsorted
from pathlib import Path
import sys
import torch_sparse
from torch_geometric.utils import add_self_loops
from torch_geometric.typing import Adj, OptTensor
from torch_scatter import scatter_add

# Constants
TRAIN_INPUT_FOLDER = "../dataset_graph_full/original_data/train"
TRAIN_OUTPUT_FOLDER = "../dataset_graph_full/original_data/npz_data/train"
TEST_INPUT_FOLDER = "../dataset_graph_full/original_data/test"
TEST_OUTPUT_FOLDER = "../dataset_graph_full/original_data/npz_data/test"
VALIDATION_DIR_INPUT = "../dataset_graph_full/original_data/npz_data/validation"
RANDOM_SEED = 1
VALIDATION_SPLIT = 0.1
MISSING_PERCENTAGE = 98
NUM_NEIGHBOURS = 8
SAVE_GRAPHS_FOLDER = "../dataset_graph_full/training_FP_noisy"

# Gaussian noise parameters
mean = 0
variance = 0.01
std_dev = np.sqrt(variance)

def add_gaussian_noise_to_features(features, mean, std_dev):
    noise = np.random.normal(mean, std_dev, (features.shape[0], 3))
    features[:, :3] += noise
    return features

# Function definitions
def get_dynamic_viscosity(temp):
    # Returns dynamic viscosity based on temperature
    T_ref = 333.15  # reference temperature [K]
    nu_ref = 1.947959242645e-4  # reference dynamic viscosity [g/cm/s]
    t = temp * T_ref
    nu = (2.46317040e-05 +
          t*(6.10895392e-07 + t*(-3.5394496e-10 + t*(1.75040791e-13 + t*(-4.5734874e-17 + 4.7456719e-21*t)))))
    nu = nu / nu_ref
    return nu


def read_vtp_slice(file_name):
    print(f"Reading VTP slice from {file_name}...")
    reader = vtk.vtkXMLPolyDataReader()
    reader.SetFileName(file_name)
    reader.Update()
    data_in = reader.GetOutput()
    temperature = np.array(data_in.GetPointData().GetArray("temperature"))
    viscosity = get_dynamic_viscosity(temperature)

    data_out = {
        'x': np.array(data_in.GetPoints().GetData())[:, 0],
        'y': np.array(data_in.GetPoints().GetData())[:, 1],
        'x_velocity': np.array(data_in.GetPointData().GetArray("x_velocity")),
        'y_velocity': np.array(data_in.GetPointData().GetArray("y_velocity")),
        'z_velocity': np.array(data_in.GetPointData().GetArray("z_velocity")),
        'pressure': np.array(data_in.GetPointData().GetArray("pressure")),
        'viscosity': viscosity
    }
    return data_out

def vtp_to_npz(input_folder, output_folder):
    print(f"Converting VTP files in {input_folder} to NPZ format...")
    
    os.makedirs(output_folder, exist_ok=True)

    vtp_files = glob.glob(os.path.join(input_folder, "*.vtp"))
    print(f"Found {len(vtp_files)} .vtp files in {input_folder}")

    for vtp_file in vtp_files:
        print(f"Processing {vtp_file}...")
        data = read_vtp_slice(vtp_file)
        output_file = os.path.join(output_folder, os.path.splitext(os.path.basename(vtp_file))[0] + ".npz")
        np.savez(output_file, **data)
        print(f"Converted {vtp_file} to {output_file}")
    print("VTP to NPZ conversion complete.")

def train_validation_split(train_dir, validation_dir):
    _, _, files = next(os.walk(train_dir))
    files = natsorted([f for f in files if f.endswith('.npz')])
    num_files = len(files)

    range_files = np.arange(num_files)
    np.random.seed(RANDOM_SEED)
    np.random.shuffle(range_files)

    validation_files = range_files[:int(num_files * VALIDATION_SPLIT)]

    # Create validation dir
    Path(validation_dir).mkdir(parents=True, exist_ok=True)

    # Move files to validation dir
    for i in validation_files:
        print(f'Moving npz file: {files[i]} from train dir to validation dir')
        src = os.path.join(train_dir, files[i])
        dst = os.path.join(validation_dir, files[i])
        shutil.move(src, dst)

def extract_random_points(data, percentage):
    print("Extracting random points from npz file")
    num_points = int(data.shape[0] * percentage)
    indices = random.sample(range(data.shape[0]), num_points)
    return np.array(indices)

def process_npz_files(folder, percentage):
    npz_files = glob.glob(os.path.join(folder, '*.npz'))
    output_folder = f'{folder}_inputs_{percentage:.0f}'  
    
    os.makedirs(output_folder, exist_ok=True)
    
    for file_path in npz_files:
        print(f"Processing npz file: {file_path}")
        with np.load(file_path) as data:
            x = data['x']
            y = data['y']
            p = data['pressure']  # Assuming that pressure is stored under the key 'pressure'
            viscosity = data['viscosity']  # Assuming viscosity is stored under the key 'viscosity'
            x_velocity = data['x_velocity']
            y_velocity = data['y_velocity']
            z_velocity = data['z_velocity']

        velocities = np.column_stack((x_velocity, y_velocity, z_velocity))
        indices_to_remove = extract_random_points(velocities, percentage / 100)  # Apply to velocities only
        velocities[indices_to_remove] = 0

        features = np.column_stack((velocities, p, viscosity, x, y))

        output_file_path = os.path.join(output_folder, os.path.basename(file_path))
        np.savez(output_file_path, x=features[:, -2], y=features[:, -1],
                 x_velocity=features[:, 0], y_velocity=features[:, 1],
                 z_velocity=features[:, 2], pressure=features[:, 3], viscosity=features[:, 4])

def load_npz_data(file_path):
    print(f"Loading data from npz file: {file_path}")
    with np.load(file_path) as data:
        x = data['x']
        y = data['y']
        p = data['pressure']
        viscosity = data['viscosity']
        x_velocity = data['x_velocity']
        y_velocity = data['y_velocity']
        z_velocity = data['z_velocity']

    # Check if first three features are zero
    indicator = ((x_velocity == 0.0) & (y_velocity == 0.0) & (z_velocity == 0.0)).astype(float)
    
    # Calculate and print the percentage of missing nodes
    missing_percentage = np.mean(indicator) * 100
    print(f"Percentage of missing nodes: {missing_percentage}%")
    
    # Assemble all the features together
    features = np.column_stack((x_velocity, y_velocity, z_velocity, p, viscosity, indicator, x, y))

    coordinates = np.column_stack((x, y))

    return torch.tensor(features, dtype=torch.float), torch.tensor(coordinates, dtype=torch.float)

def create_and_save_graph(features, coordinates, num_neighbours, folder, file_base, is_input):
    print("Creating graph from npz data")
    tree = cKDTree(coordinates.numpy())
    distances, indices = tree.query(coordinates.numpy(), k=num_neighbours+1)

    edge_index = []
    edge_attr = []  # List to store edge weights
    for v in range(len(indices)):
        for j, neighbor in enumerate(indices[v]):
            if neighbor != v:  # remove self-connections
                edge_index.append([v, neighbor])
                edge_attr.append(distances[v][j])  # Add the corresponding distance as an edge weight

    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
    edge_attr = torch.tensor(edge_attr, dtype=torch.float)  # Convert edge weights to a tensor

    # Convert to undirected graph and ensure edge attributes are also undirected
    edge_index, edge_attr = to_undirected(edge_index, edge_attr)

    graph = Data(x=features, edge_index=edge_index, edge_attr=edge_attr)

    # Only propagate features for input data, not for labels
    if is_input:

        features[:, :3] = add_gaussian_noise_to_features(features[:, :3], mean, std_dev)

        velocity_features = features[:, :3]  # Assuming velocity features are the first 3
        other_features = features[:, 3:]

        mask = velocity_features.sum(dim=-1) != 0  # Mask for the existing velocity values

        # Propagate velocity features
        model = FeaturePropagation(num_iterations=5)
        propagated_velocity_features = model.propagate(velocity_features, edge_index, mask=mask)

        # Combine propagated velocity features with other features
        propagated_features = torch.cat([propagated_velocity_features, other_features], dim=-1)
        graph.x = propagated_features

    # Save the graph
    os.makedirs(folder, exist_ok=True)
    suffix = f"_input.pt" if is_input else f"_label.pt"
    file_path = os.path.join(folder, f"{file_base}{suffix}")
    torch.save(graph, file_path)
    print(f"Saved the graph to {file_path}")



def create_graphs(data_folder, num_neighbours, save_folder, is_input):
    for i, file in enumerate(os.listdir(data_folder)):
        if file.endswith(".npz"):
            file_path = os.path.join(data_folder, file)
            file_base = os.path.splitext(file)[0]  # Remove file extension to get the base file name
            print(f"Creating graphs from file: {file_path}")
            features, coordinates = load_npz_data(file_path)
            create_and_save_graph(features, coordinates, num_neighbours, save_folder, file_base, is_input)

def get_symmetrically_normalized_adjacency(edge_index, n_nodes):
    """
    Given an edge_index, return the same edge_index and edge weights computed as
    \mathbf{\hat{D}}^{-1/2} \mathbf{\hat{A}} \mathbf{\hat{D}}^{-1/2}.
    """
    edge_weight = torch.ones((edge_index.size(1),), device=edge_index.device)
    row, col = edge_index[0], edge_index[1]
    deg = scatter_add(edge_weight, col, dim=0, dim_size=n_nodes)
    deg_inv_sqrt = deg.pow_(-0.5)
    deg_inv_sqrt.masked_fill_(deg_inv_sqrt == float("inf"), 0)
    DAD = deg_inv_sqrt[row] * edge_weight * deg_inv_sqrt[col]

    return edge_index, DAD

class FeaturePropagation(torch.nn.Module):
    def __init__(self, num_iterations: int):
        super(FeaturePropagation, self).__init__()
        self.num_iterations = num_iterations

    def propagate(self, x: Tensor, edge_index: Adj, mask: OptTensor = None) -> Tensor:
        # out is initialized to 0 for missing values. However, its initialization does not matter for the final
        # value at convergence
        out = x
        if mask is not None:
            out = torch.zeros_like(x)
            out[mask] = x[mask]

        n_nodes = x.shape[0]
        adj = self.get_propagation_matrix(out, edge_index, n_nodes)
        for _ in range(self.num_iterations):
            # Diffuse current features
            out = torch.sparse.mm(adj, out)
            # Reset original known features
            if mask is not None:
                out[mask] = x[mask]

        return out

    def get_propagation_matrix(self, x, edge_index, n_nodes):
        # Initialize all edge weights to ones if the graph is unweighted
        edge_index, edge_weight = get_symmetrically_normalized_adjacency(edge_index, n_nodes=n_nodes)
        adj = torch.sparse.FloatTensor(edge_index, values=edge_weight, size=(n_nodes, n_nodes)).to(edge_index.device)

        return adj

def propagate_features(graph, features, num_iterations):
    print("Starting feature propagation...")
    edge_index = graph.edge_index

    mask = features[:, 3] == 0  # Mask for the existing values, assuming indicator is the 4th feature

    # Propagate features using the FeaturePropagation method
    model = FeaturePropagation(num_iterations=num_iterations)
    propagated_features = model.propagate(features, edge_index, mask=mask)

    print("Feature propagation complete.")
    return propagated_features
'''
def create_graphs(data_folder, num_neighbours, save_folder, is_input):
    for i, file in enumerate(os.listdir(data_folder)):
        if file.endswith(".npz"):
            file_path = os.path.join(data_folder, file)
            file_base = os.path.splitext(file)[0]  # Remove file extension to get the base file name
            print(f"Creating graphs from file: {file_path}")
            features, coordinates = load_npz_data(file_path)
            create_and_save_graph(features, coordinates, num_neighbours, save_folder, file_base, is_input)
'''
def create_graphs(data_folder, num_neighbours, save_folder, is_input):
    for i, file in enumerate(os.listdir(data_folder)):
        # Stop after processing 5 files
        if i >= 3:
            break

        if file.endswith(".npz"):
            file_path = os.path.join(data_folder, file)
            file_base = os.path.splitext(file)[0]  # Remove file extension to get the base file name
            print(f"Creating graphs from file: {file_path}")
            features, coordinates = load_npz_data(file_path)
            create_and_save_graph(features, coordinates, num_neighbours, save_folder, file_base, is_input)

# Main script
if __name__ == "__main__":
    # Convert train and test vtp files to npz files
    #vtp_to_npz(TRAIN_INPUT_FOLDER, TRAIN_OUTPUT_FOLDER)
    #vtp_to_npz(TEST_INPUT_FOLDER, TEST_OUTPUT_FOLDER)
    # Split train data into train and validation
    #train_validation_split(TRAIN_OUTPUT_FOLDER, VALIDATION_DIR_INPUT)
    # Process npz files
    #process_npz_files(TRAIN_OUTPUT_FOLDER, MISSING_PERCENTAGE)
    #process_npz_files(TEST_OUTPUT_FOLDER, MISSING_PERCENTAGE)
    #process_npz_files(VALIDATION_DIR_INPUT, MISSING_PERCENTAGE)
    # Create and save graphs
    '''
    create_graphs(TRAIN_OUTPUT_FOLDER, NUM_NEIGHBOURS, os.path.join(SAVE_GRAPHS_FOLDER, f"train_graphs_{MISSING_PERCENTAGE}"), is_input=False)
    print("Train graphs created.")
    sys.stdout.flush()
     
    create_graphs(TRAIN_OUTPUT_FOLDER + f'_inputs_{MISSING_PERCENTAGE}', NUM_NEIGHBOURS, os.path.join(SAVE_GRAPHS_FOLDER, f"train_input_graphs_{MISSING_PERCENTAGE}"), is_input=True)
    print("Train input graphs created.")
    sys.stdout.flush()

    create_graphs(VALIDATION_DIR_INPUT + f'_inputs_{MISSING_PERCENTAGE}', NUM_NEIGHBOURS, os.path.join(SAVE_GRAPHS_FOLDER, f"validation_input_graphs_{MISSING_PERCENTAGE}"), is_input=True)
    print("Validation input graphs created.")
    sys.stdout.flush()

    create_graphs(VALIDATION_DIR_INPUT, NUM_NEIGHBOURS, os.path.join(SAVE_GRAPHS_FOLDER, f"validation_graphs_{MISSING_PERCENTAGE}"), is_input=False)  
    print("Validation graphs created.")
    sys.stdout.flush()
    
    '''
    create_graphs(TEST_OUTPUT_FOLDER + f'_inputs_{MISSING_PERCENTAGE}', NUM_NEIGHBOURS, os.path.join(SAVE_GRAPHS_FOLDER, f"test_input_graphs_{MISSING_PERCENTAGE}"), is_input=True)
    print("Test input graphs created.")
    sys.stdout.flush()
    create_graphs(TEST_OUTPUT_FOLDER, NUM_NEIGHBOURS, os.path.join(SAVE_GRAPHS_FOLDER, f"test_graphs_{MISSING_PERCENTAGE}"), is_input=False)
    print("Test graphs created.")
    sys.stdout.flush()

