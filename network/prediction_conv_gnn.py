import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
from torch.utils.data import Dataset as TorchDataset
import torch_geometric
from torch_geometric.utils import to_networkx
import networkx as nx
from multiprocessing import Pool
from pykrige.ok import OrdinaryKriging
from scipy.ndimage import gaussian_filter
from scipy.ndimage import uniform_filter
import matplotlib.gridspec as gridspec
from mpl_toolkits.axes_grid1 import make_axes_locatable
from PIL import Image

from models import (red_GAT_98_6,GAT_98_8_SkipConnections,
    GAT_98_3, GAT_98_4,GAT_98_6,
    GCN_95_8, GraphSAGE_95_8, GCN_90_6_Double, GraphSAGE_90_6_Double,
    GAT_98_10, GAT_98_8, GAT_90_6_Double, GAT_95_12,
    GAT_95_10, GAT_95_8, GraphSAGE_90_8, GATv2_90_8, GAT_90_8,
    GAT_90_8_Increased, GAT_90_3, GAT_90_3_2heads, GAT_50, GCN_50,
    GAT_90, GraphSAGE_90, GCN_90, GraphSAGE_95, GraphSAGE_99, GAT_90_6,
    GAT_90_6_2heads
)

MISSING_PERCENTAGE = 98
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CHECKPOINT_PATH = '../trained_models_FP_full/FLUID_skip_GAT_8_98_epochs_50_lr_0.0001_batch_1.pth.tar' 

def compare_positions(graph1, graph2, description):
    positions1 = graph1.x[:, -2:].cpu().numpy()
    positions2 = graph2.x[:, -2:].cpu().numpy()

    if positions1.shape != positions2.shape:
        print(f"Mismatch in the number of nodes between {description}.")
    else:
        position_difference = np.abs(positions1 - positions2)
        max_position_difference = np.max(position_difference)
        print(f"Maximum difference in node positions between {description}: {max_position_difference}")

def rmse_per_component(pred, target):

    rmse_x = torch.sqrt(torch.mean((pred[:, 0] - target[:, 0]) ** 2))
    rmse_y = torch.sqrt(torch.mean((pred[:, 1] - target[:, 1]) ** 2))

    return rmse_x.item(), rmse_y.item()


def calculate_mean_divergence(edge_index, node_features):
    """
    Calculate the mean divergence for the entire graph.

    Parameters:
    - edge_index (LongTensor): The edge indices of the graph.
    - node_features (Tensor): Node features (velocity and positions).
    
    Returns:
    - float: Mean divergence of the graph.
    """
    num_nodes = node_features.size(0)
    u, v = node_features[:, 0], node_features[:, 1]  # Velocity components
    x, y = node_features[:, 6], node_features[:, 7]  # Position components

    # Prepare tensors to store the derivatives
    du_dx = torch.zeros(num_nodes)
    dv_dy = torch.zeros(num_nodes)

    # Calculate derivatives using finite differences
    for i in range(num_nodes):
        neighbors = (edge_index[0] == i).nonzero(as_tuple=True)[0]
        neighbor_indices = edge_index[1][neighbors]

        # Handle cases with no neighbors
        if len(neighbor_indices) == 0:
            continue

        # Calculate average difference in position and velocity components
        dx = torch.mean(x[neighbor_indices] - x[i])
        dy = torch.mean(y[neighbor_indices] - y[i])

        # Avoid division by zero
        if dx != 0:
            du_dx[i] = torch.mean(u[neighbor_indices] - u[i]) / dx
        if dy != 0:
            dv_dy[i] = torch.mean(v[neighbor_indices] - v[i]) / dy

    # Calculate divergence at each node and then compute the mean
    divergence = du_dx + dv_dy
    mean_divergence = torch.mean(divergence).item()
    return mean_divergence



def get_adj_list(edge_index, num_nodes):
    """
    Convert edge_index to an adjacency list representation.
    
    Parameters:
    - edge_index (LongTensor): The edge indices.
    - num_nodes (int): Total number of nodes in the graph.
    
    Returns:
    - List[List[int]]: Adjacency list of the graph.
    """
    adj_list = [[] for _ in range(num_nodes)]
    for i, j in edge_index.t().tolist():
        adj_list[i].append(j)
        adj_list[j].append(i)  # since the graph is undirected
    return adj_list

def diffuse_graph_signal(edge_index, x, num_iterations=5, alpha=0.2):
    """
    Perform heat-based graph signal diffusion using adjacency list.
    
    Parameters:
    - edge_index (LongTensor): The edge indices.
    - x (Tensor): Node features to be diffused.
    - num_iterations (int): Number of diffusion iterations.
    - alpha (float): Diffusion coefficient. Determines the rate of diffusion.
    
    Returns:
    - Tensor: Diffused node features.
    """
    num_nodes = x.size(0)
    adj_list = get_adj_list(edge_index, num_nodes)

    for _ in range(num_iterations):
        diffusion_term = torch.zeros_like(x)
        for node, neighbors in enumerate(adj_list):
            diffusion_term[node] = x[node] - torch.mean(x[neighbors], dim=0)
        
        x = x - alpha * diffusion_term
    
    return x

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

    avg_rmse_x, avg_rmse_y = rmse_per_component(predicted_features[:, :2]*7.035423, single_graph.y[:, :2]*7.035423)

    print(f"Average RMSE for x-velocity: {avg_rmse_x}, y-velocity: {avg_rmse_y}")
    # Compare positions between output and target graphs
    print("\nComparing node positions between output and target graphs:")
    compare_positions(output_graph, label_graph, "output and target graphs")

    # Check if output is entirely zero
    print("Output zero check:", torch.all(output_graph.x == 0).item())

    print("\nComparing node positions between input and target graphs:")
    compare_positions(single_graph, label_graph, "input and target graphs")

    # Define grid size
    grid_size = 700   # Increased for a smoother plot

    # Get minimum and maximum position values
    min_x, min_y = np.min(positions[:, 0]), np.min(positions[:, 1])
    max_x, max_y = np.max(positions[:, 0]), np.max(positions[:, 1])

    # Create the grid
    grid_x, grid_y = np.mgrid[min_x:max_x:grid_size*1j, min_y:max_y:grid_size*1j]

    channels = ['x-velocity', 'y-velocity']

    target_color_ranges = []
    for i in range(2):
        target_values = single_graph.y.cpu()[:, i].numpy() * 7.035423
        color_range = (np.min(target_values), np.max(target_values))
        target_color_ranges.append(color_range)

    fig = plt.figure(figsize=(30, 10))  # Adjusted figure size for additional color bars
    gs_main = gridspec.GridSpec(2, 10, width_ratios=[1, 0.05, 1, 0.05, 1, 0.05, 1, 0.05, 1, 0.05], wspace=0.1, hspace=0.1)

    # Determine the color range for the target plot
    target_color_ranges = []
    for i in range(2):
        target_values = single_graph.y.cpu()[:, i].numpy() * 7.035423
        color_range = (np.min(target_values), np.max(target_values))
        target_color_ranges.append(color_range)

    for i in range(2):
        for j in range(4):
            plot_col_index = j * 2
            ax = fig.add_subplot(gs_main[i, plot_col_index])

            if j < 2:  # Input and Output
                values = single_graph.x.cpu()[:, i].numpy() * 7.035423 if j == 0 else output_graph.x.cpu()[:, i].numpy() * 7.035423
                vmin, vmax = target_color_ranges[i]  # Use target plot's range
            elif j == 2:  # Target
                values = single_graph.y.cpu()[:, i].numpy() * 7.035423
                vmin, vmax = target_color_ranges[i]
            elif j == 3:  # Difference
                values = output_graph.x.cpu()[:, i].numpy() - single_graph.y.cpu()[:, i].numpy()
                vmin, vmax = np.min(values), np.max(values)  # Separate range for difference plot

            grid_values = griddata(positions, values, (grid_x, grid_y), method='nearest')
            cmap = 'jet' if j < 3 else 'RdBu_r'
            im = ax.imshow(grid_values.T[::-1], extent=(min_x, max_x, min_y, max_y), origin='lower', cmap=cmap, vmin=vmin, vmax=vmax)
            ax.set_aspect('equal', adjustable='box')
            ax.set_title(r'${}$ {} m/s'.format(["Input", "Output", "Target", "Difference"][j], channels[i]))
            ax.set_xticks([])
            ax.set_yticks([])

            # Add color bar next to each plot
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="4%", pad=0.05)
            plt.colorbar(im, cax=cax)

    filename = '../results/{}_plot.png'.format(os.path.basename(CHECKPOINT_PATH).split('.')[0])
    fig.savefig(filename, dpi=600, bbox_inches='tight', pad_inches=0.2)
    img = Image.open(filename)
    img.show()
    #### commit
if __name__ == "__main__":
    #input_file = f'../dataset_graph/original_data/onehundred/test_input_graphs_50/input_interpolated_input_50.pt'
    #label_file = f'../dataset_graph/original_data/onehundred/test_label_graphs_50/label_interpolated_label_50.pt'
    #input_file = f'../dataset_graph/training_FP/test_input_graphs_{MISSING_PERCENTAGE}/cyc11_CAD618_Y18_Z0_X1_input.pt'
    #label_file = f'../dataset_graph/training_FP/test_graphs_{MISSING_PERCENTAGE}/cyc11_CAD618_Y18_Z0_X1_label.pt'
    #input_file = f'../PIV_data/test_graphs/test_input_graphs_{MISSING_PERCENTAGE}/PIV_cyc_10_CAD_625_input.pt'
    #label_file = f'../PIV_data/test_graphs/test_graphs_{MISSING_PERCENTAGE}/PIV_cyc_10_CAD_625_label.pt'
    label_file  = f'../dataset_graph_full/training_FP/test_graphs_98/cyc09_CAD635_Y8_Z1_X1_label.pt'
    input_file  = f'../dataset_graph_full/training_FP/test_input_graphs_98/cyc09_CAD635_Y8_Z1_X1_input.pt'
    run_GCN(input_file, label_file)
    