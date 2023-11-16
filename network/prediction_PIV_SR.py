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
from scipy.ndimage import gaussian_filter
from scipy.ndimage import uniform_filter

from models import (red_GAT_98_6, GAT_98_8_SkipConnections,
                    GAT_98_3, GAT_98_4, GAT_98_6,
                    GCN_95_8, GraphSAGE_95_8, GCN_90_6_Double, GraphSAGE_90_6_Double,
                    GAT_98_10, GAT_98_8, GAT_90_6_Double, GAT_95_12,
                    GAT_95_10, GAT_95_8, GraphSAGE_90_8, GATv2_90_8, GAT_90_8,
                    GAT_90_8_Increased, GAT_90_3, GAT_90_3_2heads, GAT_50, GCN_50,
                    GAT_90, GraphSAGE_90, GCN_90, GraphSAGE_95, GraphSAGE_99, GAT_90_6,
                    GAT_90_6_2heads
                    )

MISSING_PERCENTAGE = 50
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CHECKPOINT_PATH = '../trained_models_FP_full/FLUID_skip_GAT_8_98_epochs_50_lr_0.0001_batch_1.pth.tar'
UREF = 7.035423


def rmse_per_node(pred, target):
    """Computes root mean squared error per node"""

    print(pred.shape, target.shape)

    return torch.sqrt(torch.mean((pred - target) ** 2))


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


def diffuse_graph_signal(edge_index, x, num_iterations=0, alpha=0.2):
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


def run_GCN(input_file, label_file, indices_rp_file):
    # Load dataset
    test_dataset = CustomDataset([input_file], [label_file])

    # Indices of removed points
    indices_rp = np.load(indices_rp_file)

    # Select a single test graph
    single_graph = test_dataset[0]

    print("Input Graph:")
    print_graph_info(single_graph)

    # Print information about the label graph
    label_graph = torch.load(label_file)
    print("Label Graph:")
    print_graph_info(label_graph)

    # Assuming that the positions are the last 2 features in the feature vector
    positions_input = single_graph.x[:, -2:].numpy()
    positions_target = single_graph.y[:, -2:].numpy()

    # Print min and max of the positions
    print(f'Min x position: {np.min(positions_input[:, 0])}')
    print(f'Max x position: {np.max(positions_input[:, 0])}')
    print(f'Min y position: {np.min(positions_input[:, 1])}')
    print(f'Max y position: {np.max(positions_input[:, 1])}')

    ######## MODEL ########
    model = GAT_98_8_SkipConnections()
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

    # Assuming the indicator is the fourth feature (index 3)
    indicator = single_graph.x[:, 5] == 1

    # Clone the original output to create a tensor for the corrected output
    corrected_output = out.clone()

    # Manually overwrite the prediction for nodes with the indicator set to 1
    # corrected_output[~indicator, :2] = single_graph.x[~indicator, :2]

    # Now diffusing the corrected output
    diffused_corrected_output = corrected_output.clone()
    # diffused_corrected_output[:, :2] = diffuse_graph_signal(single_graph.edge_index, corrected_output[:, :2].cpu())

    # Define grid size
    grid_size_input = int(np.sqrt(len(single_graph.x)))  # Increased for a smoother plot
    grid_size_target = int(np.sqrt(len(single_graph.y)))

    # Get minimum and maximum position values
    min_x, min_y = np.min(positions_input[:, 0]), np.min(positions_input[:, 1])
    max_x, max_y = np.max(positions_input[:, 0]), np.max(positions_input[:, 1])

    # Create the grid
    grid_x_input, grid_y_input = np.mgrid[min_x:max_x:grid_size_input * 1j, min_y:max_y:grid_size_input * 1j]
    grid_x_target, grid_y_target = np.mgrid[min_x:max_x:grid_size_target * 1j, min_y:max_y:grid_size_target * 1j]

    fig, axs = plt.subplots(3, 2, figsize=(10, 10))  # Changed the subplot configuration
    fig.subplots_adjust(hspace=0.5, wspace=0.5)

    # Variables to keep track of min and max difference across all channels
    # diff_min = np.inf
    # diff_max = -np.inf

    channels = ['x-velocity', 'y-velocity']

    for i in range(2):
        # scaled_output_values = diffused_corrected_output.cpu()[:, i] * UREF
        # scaled_target_values = single_graph.y.cpu()[:, i] * UREF

        # node_wise_rmse = rmse_per_node(scaled_output_values, scaled_target_values)
        # print(f"Node-wise RMSE for channel {i}: {node_wise_rmse}")

        input_values = single_graph.x.cpu()[:, i].numpy() * UREF
        output_values = diffused_corrected_output.cpu()[:, i].numpy() * UREF
        target_values = single_graph.y.cpu()[:, i].numpy() * UREF

        # Replace the current interpolation method with griddata
        grid_input_values = griddata(positions_input, input_values, (grid_x_input, grid_y_input), method='nearest')
        grid_output_values = griddata(positions_input, output_values, (grid_x_input, grid_y_input), method='nearest')

        grid_target_values = griddata(positions_target, target_values, (grid_x_target, grid_y_target), method='nearest')

        vmin, vmax = target_values.min(), target_values.max()
        # Node wise prediction metrics
        node_wise_rmse = rmse(out[indices_rp, :], single_graph.y[indices_rp, :])
        print(f"Node-wise RMSE for {channels[i]}: {node_wise_rmse}")
        node_wise_mae = mae(out[indices_rp, :], single_graph.y[indices_rp, :])
        print(f"Node-wise MAE for {channels[i]}: {node_wise_mae}")

        im = axs[0, i].imshow(grid_input_values.T[::-1], extent=(min_x, max_x, min_y, max_y), origin='lower',
                              cmap='jet', vmin=vmin, vmax=vmax)
        axs[0, i].set_title(f'Input {channels[i]}')
        axs[0, i].set_xticks([])
        axs[0, i].set_yticks([])
        cbar1 = fig.colorbar(im, ax=axs[0, i])
        cbar1.ax.tick_params(labelsize=8)

        im = axs[1, i].imshow(grid_output_values.T[::-1], extent=(min_x, max_x, min_y, max_y), origin='lower',
                              cmap='jet', vmin=vmin, vmax=vmax)
        axs[1, i].set_title(f'Output {channels[i]}')
        axs[1, i].set_xticks([])
        axs[1, i].set_yticks([])
        cbar2 = fig.colorbar(im, ax=axs[1, i])
        cbar2.ax.tick_params(labelsize=8)

        im = axs[2, i].imshow(grid_target_values.T[::-1], extent=(min_x, max_x, min_y, max_y), origin='lower',
                              cmap='jet', vmin=vmin, vmax=vmax)
        axs[2, i].set_title(f'Target {channels[i]}')
        axs[2, i].set_xticks([])
        axs[2, i].set_yticks([])
        cbar3 = fig.colorbar(im, ax=axs[2, i])
        cbar3.ax.tick_params(labelsize=8)

        # diff_min = min(diff_min, np.min(grid_output_values - grid_target_values))
        # diff_max = max(diff_max, np.max(grid_output_values - grid_target_values))

    plt.tight_layout()
    fig.savefig('../results/{}_plot_SR.png'.format(os.path.basename(CHECKPOINT_PATH).split('.')[0]), dpi=600)
    plt.show()


if __name__ == "__main__":
    # Files
    input_file = f'../PIV_data/test_graphs/test_input_graphs_{MISSING_PERCENTAGE}/PIV_cyc_10_CAD_625_input_SR.pt'
    label_file = f'../PIV_data/test_graphs/test_graphs_{MISSING_PERCENTAGE}/PIV_cyc_10_CAD_625_label_SR.pt'
    indices_rp_file = f'../PIV_data/labels_npz_inputs_{MISSING_PERCENTAGE}/PIV_cyc_10_CAD_625_indices_rp.npy'

    run_GCN(input_file, label_file, indices_rp_file)
