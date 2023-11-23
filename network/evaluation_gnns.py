import os
import torch
import numpy as np
from torch.utils.data import Dataset as TorchDataset
from models import (red_GAT_98_6,GAT_98_8_SkipConnections,
    GAT_98_3, GAT_98_4,GAT_98_6,
    GCN_95_8, GraphSAGE_95_8, GCN_90_6_Double, GraphSAGE_90_6_Double,
    GAT_98_10, GAT_98_8, GAT_90_6_Double, GAT_95_12,
    GAT_95_10, GAT_95_8, GraphSAGE_90_8, GATv2_90_8, GAT_90_8,
    GAT_90_8_Increased, GAT_90_3, GAT_90_3_2heads, GAT_50, GCN_50,
    GAT_90, GraphSAGE_90, GCN_90, GraphSAGE_95, GraphSAGE_99, GAT_90_6,
    GAT_90_6_2heads
)
from torch_geometric.utils import to_networkx
import networkx as nx
from collections import defaultdict
import glob
import matplotlib.pyplot as plt
import csv

MISSING_PERCENTAGE = 98
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CHECKPOINT_PATH = '../trained_models_FP_full/FLUID_skip_GAT_8_98_epochs_50_lr_0.0001_batch_1.pth.tar' 

def mae_per_component(pred, target):
    """
    Computes the node-wise MAE for each component, averaged over all nodes.
    """
    # Calculate absolute differences for each node
    abs_diff_x = torch.abs(pred[:, 0] - target[:, 0])
    abs_diff_y = torch.abs(pred[:, 1] - target[:, 1])

    # Average over all nodes
    mae_x = torch.mean(abs_diff_x)
    mae_y = torch.mean(abs_diff_y)

    return mae_x.item(), mae_y.item()

def compare_positions(graph1, graph2, description):
    positions1 = graph1.x[:, -2:].cpu().numpy()
    positions2 = graph2.x[:, -2:].cpu().numpy()

    if positions1.shape != positions2.shape:
        print(f"Mismatch in the number of nodes between {description}.")
    else:
        position_difference = np.abs(positions1 - positions2)
        max_position_difference = np.max(position_difference)
        print(f"Maximum difference in node positions between {description}: {max_position_difference}")

def print_graph_info(graph):
    print("Graph Information:")
    print("------------------")
    
    print("Number of Nodes:", graph.num_nodes)
    print("Number of Edges:", graph.num_edges)
    print("Number of Node Features:", graph.num_node_features)
    print("Number of Edge Features:", graph.num_edge_features)
    print("Is Directed:", graph.is_directed())
    print("Contains Isolated Nodes:", graph.has_isolated_nodes())
    print("Contains Self-loops:", graph.has_self_loops())
    print("Is Undirected:", graph.is_undirected())

    # Convert the graph to a networkx graph for additional analysis
    g = to_networkx(graph, to_undirected=True)
    
    # Degree Distribution
    degrees = [g.degree(n) for n in g.nodes()]
    print("Average Degree:", np.mean(degrees))
    print("Minimum Degree:", np.min(degrees))
    print("Maximum Degree:", np.max(degrees))

    # Check if the graph is connected
    print("Is Connected:", nx.is_connected(g))

    # Get the number of connected components
    print("Number of Connected Components:", nx.number_connected_components(g))

    # Percentage of the first three features which are set to zero
    first_three_features_zero = torch.norm(graph.x[:, :3], p=2, dim=1) == 0
    percentage_zero = torch.mean(first_three_features_zero.float()) * 100
    print(f"Percentage of the first three features set to zero: {percentage_zero.item():.2f}%")

class CustomDataset(TorchDataset):
    def __init__(self, input_files, label_files):
        self.input_files = input_files
        self.label_files = label_files

    def __len__(self):
        return len(self.input_files)

    def __getitem__(self, idx):
        input_data = torch.load(self.input_files[idx])
        label_data = torch.load(self.label_files[idx])
        input_data.y = label_data.x
        input_data.x_complete = label_data.x
        return input_data


def load_checkpoint(model, checkpoint_path):
    checkpoint = torch.load(checkpoint_path, map_location=DEVICE)
    state_dict = checkpoint['state_dict']
    
    # Remove the "module." prefix
    new_state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
    
    model.load_state_dict(new_state_dict)

def rmse_per_component(pred, target):

    rmse_x = torch.sqrt(torch.mean((pred[:, 0] - target[:, 0]) ** 2))
    rmse_y = torch.sqrt(torch.mean((pred[:, 1] - target[:, 1]) ** 2))

    return rmse_x.item(), rmse_y.item()


def mae(pred, target):
    """Computes mean absolute error"""
    return torch.mean(torch.abs(pred - target))


def run_GCN(input_file, label_file):
    # Load dataset
    test_dataset = CustomDataset([input_file], [label_file])

    # Select a single test graph
    single_graph = test_dataset[0]

    print("Input Graph:")
    print_graph_info(single_graph)
    
    # Print information about the label graph
    label_graph = torch.load(label_file)
    print("Label Graph:")
    print_graph_info(label_graph)

    # Assuming that the positions are the last 2 features in the feature vector
    positions = single_graph.x[:, -2:].numpy()

    model = GAT_98_8_SkipConnections()
    model.to(DEVICE)
    load_checkpoint(model, CHECKPOINT_PATH)

    model.eval()

    # Save original position features
    original_positions = single_graph.x[:, -2:].clone()

    # Move data to the correct device
    single_graph.to(DEVICE)

    # Perform prediction
    with torch.no_grad():
        predicted_features = model(single_graph)

    # Replace last two features of the output with original position features
    predicted_features[:, -2:] = original_positions.to(DEVICE)

    # Create a new graph for output comparison
    output_graph = single_graph.clone()
    output_graph.x = predicted_features


    print("\nComparing node positions between output and target graphs:")
    compare_positions(output_graph, label_graph, "output and target graphs")

    # Check if output is entirely zero
    print("Output zero check:", torch.all(output_graph.x == 0).item())

    print("\nComparing node positions between input and target graphs:")
    compare_positions(single_graph, label_graph, "input and target graphs")

    rmse_values, mae_values = [], []

    target_color_ranges = []
    for i in range(2):
        target_values = single_graph.y.cpu()[:, i].numpy() * 7.035423
        color_range = (np.min(target_values), np.max(target_values))
        target_color_ranges.append(color_range)

        # Calculate RMSE and MAE for both components
        rmse_x, rmse_y = rmse_per_component(predicted_features[:, :2]*7.035423, single_graph.y[:, :2]*7.035423)
        mae_x, mae_y = mae_per_component(predicted_features[:, :2]*7.035423, single_graph.y[:, :2]*7.035423)

        # Append the values to the respective lists
        rmse_values.append(rmse_x)
        rmse_values.append(rmse_y)
        mae_values.append(mae_x)
        mae_values.append(mae_y)

        # Corrected print statement
        print(f"Component {'x-velocity' if i == 0 else 'y-velocity'}: RMSE = {rmse_x if i == 0 else rmse_y}, MAE = {mae_x if i == 0 else mae_y}")

    return rmse_values, mae_values

if __name__ == "__main__":

    stats = {
        'rmse': defaultdict(list),
        'mae': defaultdict(list)
            }
    
    print(f"Missing Data Percentage: {MISSING_PERCENTAGE}%")
    print(f"Checkpoint Name: {os.path.basename(CHECKPOINT_PATH)}")
    print("---------------------------------------------------")
    input_folder = f'../dataset_graph_full/training_FP/test_input_graphs_{MISSING_PERCENTAGE}'
    label_folder = f'../dataset_graph_full/training_FP/test_graphs_{MISSING_PERCENTAGE}'

    input_files = glob.glob(f"{input_folder}/*.pt")
    label_files = glob.glob(f"{label_folder}/*.pt")

    input_files.sort()
    label_files.sort()

    rmse_accumulator = defaultdict(list)
    mae_accumulator = defaultdict(list)
    results = []

    for input_file, label_file in zip(input_files, label_files):
        print(f"Analyzing input file: {input_file} and label file: {label_file}")
        rmse_values, mae_values = run_GCN(input_file, label_file)

        results.append((rmse_values[0], rmse_values[1], mae_values[0], mae_values[1]))


    avg_rmse = {i: sum(vals) / len(vals) for i, vals in rmse_accumulator.items()}
    avg_mae = {i: sum(vals) / len(vals) for i, vals in mae_accumulator.items()}

    for component in avg_rmse:
        component_name = 'x-velocity' if component % 2 == 0 else 'y-velocity'
        print(f"Average RMSE for {component_name}: {avg_rmse[component]:.4f}")
        print(f"Average MAE for {component_name}: {avg_mae[component]:.4f}")

    # Write the results to a CSV file
    with open('velocity_gnn.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['RMSE X Velocity', 'RMSE Y Velocity', 'MAE X Velocity', 'MAE Y Velocity'])
        writer.writerows(results)

    print("RMSE and MAE values for each input slice saved to 'velocity_errors_gnn.csv'")
