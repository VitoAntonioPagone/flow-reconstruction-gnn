import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
from torch.utils.data import Dataset as TorchDataset
from models import GAT_50, GCN_50, AGNN_90, GAT_95, GraphSAGE_95, GraphSAGE_99, Hole_GraphSAGE_99, Hole_GAT_99, Hole_GCN_90
import torch_geometric
from torch_geometric.utils import to_networkx
import networkx as nx
from collections import defaultdict
import glob

MISSING_PERCENTAGE = 99
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CHECKPOINT_PATH = '../trained_models/GCN13_box_90.0_epochs_100_lr_0.0001_batch_4.pth.tar' 

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



def rmse(pred, target):
    """Computes root mean squared error"""
    return torch.sqrt(torch.mean((pred - target) ** 2))


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

    # Print min and max of the positions
    print(f'Min x position: {np.min(positions[:, 0])}')
    print(f'Max x position: {np.max(positions[:, 0])}')
    print(f'Min y position: {np.min(positions[:, 1])}')
    print(f'Max y position: {np.max(positions[:, 1])}')

    ######## MODEL ########
    model = Hole_GCN_90()
    ######## MODEL ########
    
    model.to(DEVICE)

    # Load trained weights
    load_checkpoint(model, CHECKPOINT_PATH)

    model.eval()

    # Move data to the correct device
    single_graph.x = single_graph.x.to(DEVICE)
    single_graph.edge_index = single_graph.edge_index.to(DEVICE)
    single_graph.y = single_graph.y.to(DEVICE)

    # Perform prediction
    with torch.no_grad():
        out = model(single_graph)

    # Update only nodes where the fourth feature is set to 1.0
    indicator_nodes = single_graph.x[:, 3] == 1.0
    out[~indicator_nodes, :3] = single_graph.x[~indicator_nodes, :3]


    # Check if output is entirely zero
    print("Output zero check:", torch.all(out == 0).item())

    # Define grid size
    grid_size = 256   # Increased for a smoother plot

    # Get minimum and maximum position values
    min_x, min_y = np.min(positions[:, 0]), np.min(positions[:, 1])
    max_x, max_y = np.max(positions[:, 0]), np.max(positions[:, 1])

    # Create the grid
    grid_x, grid_y = np.mgrid[min_x:max_x:grid_size*1j, min_y:max_y:grid_size*1j]

    def rmse(pred, target):
        """Computes root mean squared error"""
        return torch.sqrt(torch.mean((pred - target) ** 2))

    fig, axs = plt.subplots(4, 3, figsize=(10, 10))  # Changed the subplot configuration
    fig.subplots_adjust(hspace=0.5, wspace=0.5) 
    
    def mae(pred, target):
        """Computes mean absolute error"""
        return torch.mean(torch.abs(pred - target))

    # Variables to keep track of min and max difference across all channels
    diff_min = np.inf
    diff_max = -np.inf

    channels = ['x-velocity', 'y-velocity', 'z-velocity']
    rmse_values = []
    mae_values = []
    for i in range(3):
        input_values = single_graph.x.cpu()[:, i].numpy() * 7.035423
        output_values = out.cpu()[:, i].numpy() * 7.035423
        target_values = single_graph.y.cpu()[:, i].numpy() * 7.035423
        grid_input_values  = griddata(positions, input_values, (grid_x, grid_y), method='nearest')
        grid_output_values = griddata(positions, output_values, (grid_x, grid_y), method='nearest')
        grid_target_values = griddata(positions, target_values, (grid_x, grid_y), method='nearest')
        vmin, vmax = target_values.min(), target_values.max()
        pixel_wise_rmse = rmse(torch.tensor(grid_output_values), torch.tensor(grid_target_values))
        print(f"Pixel-wise RMSE for {channels[i]}: {pixel_wise_rmse}")
        pixel_wise_mae = mae(torch.tensor(grid_output_values), torch.tensor(grid_target_values))
        print(f"Pixel-wise MAE for {channels[i]}: {pixel_wise_mae}")
        
        pixel_wise_rmse = rmse(torch.tensor(grid_output_values), torch.tensor(grid_target_values))
        rmse_values.append(pixel_wise_rmse.item())
        pixel_wise_mae = mae(torch.tensor(grid_output_values), torch.tensor(grid_target_values))
        mae_values.append(pixel_wise_mae.item())

        im = axs[0, i].imshow(grid_input_values.T[::-1], extent=(min_x, max_x, min_y, max_y), origin='lower', cmap='jet', vmin=vmin, vmax=vmax)
        axs[0, i].set_title(f'Input {channels[i]}')
        axs[0, i].set_xticks([])
        axs[0, i].set_yticks([])
        cbar1 = fig.colorbar(im, ax=axs[0, i])
        cbar1.ax.tick_params(labelsize=8)

        im = axs[1, i].imshow(grid_output_values.T[::-1], extent=(min_x, max_x, min_y, max_y), origin='lower', cmap='jet', vmin=vmin, vmax=vmax)
        axs[1, i].set_title(f'Output {channels[i]}')
        axs[1, i].set_xticks([])
        axs[1, i].set_yticks([])
        cbar2 = fig.colorbar(im, ax=axs[1, i])
        cbar2.ax.tick_params(labelsize=8)

        im = axs[2, i].imshow(grid_target_values.T[::-1], extent=(min_x, max_x, min_y, max_y), origin='lower', cmap='jet', vmin=vmin, vmax=vmax)
        axs[2, i].set_title(f'Target {channels[i]}')
        axs[2, i].set_xticks([])
        axs[2, i].set_yticks([])
        cbar3 = fig.colorbar(im, ax=axs[2, i])
        cbar3.ax.tick_params(labelsize=8)

        diff_min = min(diff_min, np.min(grid_output_values - grid_target_values))
        diff_max = max(diff_max, np.max(grid_output_values - grid_target_values))

    for i in range(3):
        # Use the scaled data for calculating the difference
        scaled_output_values = out.cpu()[:, i].numpy() * 7.035423
        scaled_target_values = single_graph.y.cpu()[:, i].numpy() * 7.035423

        # Calculate the difference using scaled data
        diff_values = scaled_output_values - scaled_target_values

        # Then create the grid for this difference
        grid_diff_values = griddata(positions, diff_values, (grid_x, grid_y), method='nearest')

        im = axs[3, i].imshow(grid_diff_values.T[::-1], extent=(min_x, max_x, min_y, max_y), origin='lower', cmap='jet', vmin=diff_min, vmax=diff_max)
        axs[3, i].set_title(f'Difference {channels[i]}')
        axs[3, i].set_xticks([])
        axs[3, i].set_yticks([])
        cbar4 = fig.colorbar(im, ax=axs[3, i], orientation='vertical')
        cbar4.ax.tick_params(labelsize=8)

    return rmse_values, mae_values

if __name__ == "__main__":
    # Specify the paths to the two folders containing the input and label files
    input_folder = '../dataset_graph/training_hole/test_input_graphs_box_90.0'
    label_folder = '../dataset_graph/training_hole/test_graphs_box_90.0'


    input_files = glob.glob(f"{input_folder}/*.pt")
    label_files = glob.glob(f"{label_folder}/*.pt")


    input_files.sort()
    label_files.sort()

    rmse_accumulator = defaultdict(list)
    mae_accumulator = defaultdict(list)
    for input_file, label_file in zip(input_files, label_files):
        print(f"Analyzing input file: {input_file} and label file: {label_file}")
        rmse_values, mae_values = run_GCN(input_file, label_file)
        for i, (rmse_val, mae_val) in enumerate(zip(rmse_values, mae_values)):
            rmse_accumulator[i].append(rmse_val)
            mae_accumulator[i].append(mae_val)


    # Compute average RMSE and MAE values
    avg_rmse = {i: sum(vals) / len(vals) for i, vals in rmse_accumulator.items()}
    avg_mae = {i: sum(vals) / len(vals) for i, vals in mae_accumulator.items()}
    print("Average RMSE:", avg_rmse)
    print("Average MAE:", avg_mae)