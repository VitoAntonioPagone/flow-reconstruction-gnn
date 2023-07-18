import os
import glob
import shutil
import numpy as np
import vtk
import random
import torch
import torch_geometric
from torch_geometric.data import Data
from torch_geometric.utils import to_undirected
from scipy.spatial import cKDTree
from natsort import natsorted
from pathlib import Path
import sys

# Constants
TRAIN_INPUT_FOLDER = "../dataset_graph/original_data/train"
TRAIN_OUTPUT_FOLDER = "../dataset_graph/original_data/npz_data/train"
TEST_INPUT_FOLDER = "../dataset_graph/original_data/test"
TEST_OUTPUT_FOLDER = "../dataset_graph/original_data/npz_data/test"
VALIDATION_DIR_INPUT = "../dataset_graph/original_data/npz_data/validation"
RANDOM_SEED = 1
VALIDATION_SPLIT = 0.1
MISSING_PERCENTAGE = 99
NUM_NEIGHBOURS = 8
BOX_PERCENTAGE = 0.9 
SAVE_GRAPHS_FOLDER = f"../dataset_graph/training_hole"

# Make sure folders exist
os.makedirs(TRAIN_INPUT_FOLDER, exist_ok=True)
os.makedirs(TRAIN_OUTPUT_FOLDER, exist_ok=True)
os.makedirs(TEST_INPUT_FOLDER, exist_ok=True)
os.makedirs(TEST_OUTPUT_FOLDER, exist_ok=True)
os.makedirs(VALIDATION_DIR_INPUT, exist_ok=True)
os.makedirs(SAVE_GRAPHS_FOLDER, exist_ok=True)

# Function definitions
def read_vtp_slice(file_name):
    print(f"Reading VTP slice from {file_name}...")
    reader = vtk.vtkXMLPolyDataReader()
    reader.SetFileName(file_name)
    reader.Update()
    data_in = reader.GetOutput()

    data_out = {
        'x': np.array(data_in.GetPoints().GetData())[:, 0],
        'y': np.array(data_in.GetPoints().GetData())[:, 1],
        'x_velocity': np.array(data_in.GetPointData().GetArray("x_velocity")),
        'y_velocity': np.array(data_in.GetPointData().GetArray("y_velocity")),
        'z_velocity': np.array(data_in.GetPointData().GetArray("z_velocity")),
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

    # if validation dir already exists, delete it and create a new one
    if os.path.exists(validation_dir):
        shutil.rmtree(validation_dir)
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
def process_npz_files(folder, box_percentage):
    npz_files = glob.glob(os.path.join(folder, '*.npz'))
    output_folder = f'{folder}_modified_{BOX_PERCENTAGE * 100}'  # Folder name now includes box percentage
    
    # if folder already exists, delete it and create a new one
    if os.path.exists(output_folder):
        shutil.rmtree(output_folder)
    os.makedirs(output_folder, exist_ok=True)
    
    for file_path in npz_files:
        print(f"Processing npz file: {file_path}")
        with np.load(file_path) as data:
            x = data['x']
            y = data['y']
            x_velocity = data['x_velocity']
            y_velocity = data['y_velocity']
            z_velocity = data['z_velocity']

        features = np.column_stack((x, y, x_velocity, y_velocity, z_velocity))
        
        # Calculate the center of the x-y plane
        x_center = (np.max(x) - np.min(x)) / 2
        y_center = (np.max(y) - np.min(y)) / 2

        # Determine the quantiles for x and y separately
        x_min_box = np.quantile(x, (1 - box_percentage) / 2)
        x_max_box = np.quantile(x, 1 - (1 - box_percentage) / 2)
        y_min_box = np.quantile(y, (1 - box_percentage) / 2)
        y_max_box = np.quantile(y, 1 - (1 - box_percentage) / 2)
        
        # Identify the points outside the box
        outside_indices = np.where((x < x_min_box) | (x > x_max_box) | (y < y_min_box) | (y > y_max_box))
        
        # Create a new copy of the features and modify it
        modified_features = features.copy()
        modified_features[outside_indices, 2:5] = 0

        output_file_path = os.path.join(output_folder, os.path.basename(file_path))
        np.savez(output_file_path, x=modified_features[:, 0], y=modified_features[:, 1],
                 x_velocity=modified_features[:, 2], y_velocity=modified_features[:, 3],
                 z_velocity=modified_features[:, 4])


def load_npz_data(file_path):
    print(f"Loading data from npz file: {file_path}")
    with np.load(file_path) as data:
        x = data['x']
        y = data['y']
        x_velocity = data['x_velocity']
        y_velocity = data['y_velocity']
        z_velocity = data['z_velocity']

    # Check if first three features are zero
    indicator = ((x_velocity == 0.0) & (y_velocity == 0.0) & (z_velocity == 0.0)).astype(float)
    
    # Calculate and print the percentage of missing nodes
    missing_percentage = np.mean(indicator) * 100
    print(f"Percentage of missing nodes: {missing_percentage}%")
    
    # Assemble all the features together
    features = np.column_stack((x_velocity, y_velocity, z_velocity, indicator, x, y))  # Added indicator feature as the 4th feature

    coordinates = np.column_stack((x, y))

    return torch.tensor(features, dtype=torch.float), torch.tensor(coordinates, dtype=torch.float)


def simple_feature_propagation(graph, features, num_iterations=40):
    """Propagates features through the graph by setting each node's feature to be the average of its neighbors' features."""
    print("Starting feature propagation...")
    # Get the indicators of the missing nodes and known nodes
    missing_nodes = features[:, 3] == 1.0

    adjacency_matrix = torch_geometric.utils.to_dense_adj(graph.edge_index).squeeze(0)

    for i in range(num_iterations):
        print(f"Iteration {i + 1} of feature propagation...")
        for node in range(features.shape[0]):
            if missing_nodes[node]:
                neighbors = adjacency_matrix[node, :].nonzero(as_tuple=True)[0]
                features[node, :3] = torch.mean(features[neighbors, :3], dim=0)

        print(f"Finished iteration {i + 1} of feature propagation. The current feature values are:\n{features}")

    print("Feature propagation complete.")
    return features

def create_and_save_graph(features, coordinates, num_neighbours, folder, file_base, is_input):
    print("Creating graph from npz data")
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

    # Propagate features
    propagated_features = simple_feature_propagation(graph, features, num_iterations=5)
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

# Main script
if __name__ == "__main__":
    # Convert train and test vtp files to npz files
    #vtp_to_npz(TRAIN_INPUT_FOLDER, TRAIN_OUTPUT_FOLDER)
    #vtp_to_npz(TEST_INPUT_FOLDER, TEST_OUTPUT_FOLDER)
    # Split train data into train and validation
    #train_validation_split(TRAIN_OUTPUT_FOLDER, VALIDATION_DIR_INPUT)
    # Process npz files
  
    #process_npz_files(TRAIN_OUTPUT_FOLDER, BOX_PERCENTAGE)
    process_npz_files(TEST_OUTPUT_FOLDER, BOX_PERCENTAGE)
    #process_npz_files(VALIDATION_DIR_INPUT, BOX_PERCENTAGE)
    # Create and save graphs
    #create_graphs(VALIDATION_DIR_INPUT + f'_modified_{BOX_PERCENTAGE * 100}', NUM_NEIGHBOURS, os.path.join(SAVE_GRAPHS_FOLDER, f"validation_input_graphs_box_{BOX_PERCENTAGE * 100}"), is_input=True)
    #print("Validation input graphs created.")
    #sys.stdout.flush()
    #create_graphs(VALIDATION_DIR_INPUT, NUM_NEIGHBOURS, os.path.join(SAVE_GRAPHS_FOLDER, f"validation_graphs_box_{BOX_PERCENTAGE * 100}"), is_input=False)
    #print("Validation graphs created.")
    #sys.stdout.flush()
    #create_graphs(TRAIN_OUTPUT_FOLDER + f'_modified_{BOX_PERCENTAGE * 100}', NUM_NEIGHBOURS, os.path.join(SAVE_GRAPHS_FOLDER, f"train_input_graphs_box_{BOX_PERCENTAGE * 100}"), is_input=True)
    #print("Train input graphs created.")
    #sys.stdout.flush()
    #create_graphs(TRAIN_OUTPUT_FOLDER, NUM_NEIGHBOURS, os.path.join(SAVE_GRAPHS_FOLDER, f"train_graphs_box_{BOX_PERCENTAGE * 100}"), is_input=False)
    #print("Train graphs created.")
    #sys.stdout.flush()
    create_graphs(TEST_OUTPUT_FOLDER + f'_modified_{BOX_PERCENTAGE * 100}', NUM_NEIGHBOURS, os.path.join(SAVE_GRAPHS_FOLDER, f"test_input_graphs_box_{BOX_PERCENTAGE * 100}"), is_input=True)
    print("Test input graphs created.")
    create_graphs(TEST_OUTPUT_FOLDER, NUM_NEIGHBOURS, os.path.join(SAVE_GRAPHS_FOLDER, f"test_graphs_box_{BOX_PERCENTAGE * 100}"), is_input=False)
    print("Test graphs created.")