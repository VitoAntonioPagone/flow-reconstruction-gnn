import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
from torch.utils.data import Dataset as TorchDataset
from torch_geometric.utils import to_networkx
import networkx as nx
from multiprocessing import Pool
from pykrige.ok import OrdinaryKriging
from scipy.ndimage import gaussian_filter
from scipy.ndimage import uniform_filter
import matplotlib.gridspec as gridspec
from mpl_toolkits.axes_grid1 import make_axes_locatable
from PIL import Image
import scienceplots
from flow_reconstruction.GACN.graph_models import GAT_98_8_SkipConnections

plt.style.use('science')

MISSING_PERCENTAGE = 98
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CHECKPOINT_PATH = '../trained_models_FP_full/FLUID_skip_GAT_8_98_epochs_50_lr_0.0001_batch_1.pth.tar' 

def compare_positions(graph1, graph2, description):
    """Compare node positions between two graphs."""
    positions1 = graph1.x[:, -2:].cpu().numpy()
    positions2 = graph2.x[:, -2:].cpu().numpy()
    if positions1.shape != positions2.shape:
        print(f"Mismatch in the number of nodes between {description}.")
    else:
        position_difference = np.abs(positions1 - positions2)
        max_position_difference = np.max(position_difference)
        print(f"Maximum difference in node positions between {description}: {max_position_difference}")

def mae_per_component(pred, target):
    """Compute MAE per velocity component."""
    abs_diff_x = torch.abs(pred[:, 0] - target[:, 0])
    abs_diff_y = torch.abs(pred[:, 1] - target[:, 1])
    mae_x = torch.mean(abs_diff_x)
    mae_y = torch.mean(abs_diff_y)
    return mae_x.item(), mae_y.item()

def rmse_per_component(pred, target):
    """Compute RMSE per velocity component."""
    squared_diff_x = (pred[:, 0] - target[:, 0]) ** 2
    squared_diff_y = (pred[:, 1] - target[:, 1]) ** 2
    rmse_x = torch.sqrt(torch.mean(squared_diff_x))
    rmse_y = torch.sqrt(torch.mean(squared_diff_y))
    return rmse_x.item(), rmse_y.item()

def calculate_mean_divergence(edge_index, node_features):
    """Calculate mean divergence for graph."""
    num_nodes = node_features.size(0)
    u, v = node_features[:, 0], node_features[:, 1]
    x, y = node_features[:, 6], node_features[:, 7]
    du_dx = torch.zeros(num_nodes)
    dv_dy = torch.zeros(num_nodes)
    for i in range(num_nodes):
        neighbors = (edge_index[0] == i).nonzero(as_tuple=True)[0]
        neighbor_indices = edge_index[1][neighbors]
        if len(neighbor_indices) == 0:
            continue
        dx = torch.mean(x[neighbor_indices] - x[i])
        dy = torch.mean(y[neighbor_indices] - y[i])
        if dx != 0:
            du_dx[i] = torch.mean(u[neighbor_indices] - u[i]) / dx
        if dy != 0:
            dv_dy[i] = torch.mean(v[neighbor_indices] - v[i]) / dy
    divergence = du_dx + dv_dy
    mean_divergence = torch.mean(divergence).item()
    return mean_divergence

def get_adj_list(edge_index, num_nodes):
    """Convert edge_index to adjacency list."""
    adj_list = [[] for _ in range(num_nodes)]
    for i, j in edge_index.t().tolist():
        adj_list[i].append(j)
        adj_list[j].append(i)
    return adj_list

def diffuse_graph_signal(edge_index, x, num_iterations=5, alpha=0.2):
    """Perform graph signal diffusion."""
    num_nodes = x.size(0)
    adj_list = get_adj_list(edge_index, num_nodes)
    for _ in range(num_iterations):
        diffusion_term = torch.zeros_like(x)
        for node, neighbors in enumerate(adj_list):
            diffusion_term[node] = x[node] - torch.mean(x[neighbors], dim=0)
        x = x - alpha * diffusion_term
    return x

def print_graph_info(graph):
    """Print graph information."""
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
    g = to_networkx(graph, to_undirected=True)
    degrees = [g.degree(n) for n in g.nodes()]
    print("Average Degree:", np.mean(degrees))
    print("Minimum Degree:", np.min(degrees))
    print("Maximum Degree:", np.max(degrees))
    print("Is Connected:", nx.is_connected(g))
    print("Number of Connected Components:", nx.number_connected_components(g))
    first_three_features_zero = torch.norm(graph.x[:, :3], p=2, dim=1) == 0
    percentage_zero = torch.mean(first_three_features_zero.float()) * 100
    print(f"Percentage of the first three features set to zero: {percentage_zero.item():.2f}%")

class CustomDataset(TorchDataset):
    """Dataset for graph input-label pairs."""
    
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
    """Load model checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location=DEVICE)
    state_dict = checkpoint['state_dict']
    new_state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
    model.load_state_dict(new_state_dict)

def rmse(pred, target):
    """Compute RMSE."""
    return torch.sqrt(torch.mean((pred - target) ** 2))

def mae(pred, target):
    """Compute MAE."""
    return torch.mean(torch.abs(pred - target))

def run_GCN(input_file, label_file):
    """Run GCN prediction and visualization."""
    test_dataset = CustomDataset([input_file], [label_file])
    single_graph = test_dataset[0]
    print("Input Graph:")
    print_graph_info(single_graph)
    label_graph = torch.load(label_file)
    print("Label Graph:")
    print_graph_info(label_graph)
    positions = single_graph.x[:, -2:].numpy()
    model = GAT_98_8_SkipConnections()
    model.to(DEVICE)
    load_checkpoint(model, CHECKPOINT_PATH)
    model.eval()
    original_positions = single_graph.x[:, -2:].clone()
    single_graph.to(DEVICE)
    with torch.no_grad():
        predicted_features = model(single_graph)
    predicted_features[:, -2:] = original_positions.to(DEVICE)
    output_graph = single_graph.clone()
    output_graph.x = predicted_features
    print("\nComparing node positions between output and target graphs:")
    compare_positions(output_graph, label_graph, "output and target graphs")
    print("Output zero check:", torch.all(output_graph.x == 0).item())
    print("\nComparing node positions between input and target graphs:")
    compare_positions(single_graph, label_graph, "input and target graphs")
    grid_size = 700   
    min_x, min_y = np.min(positions[:, 0]), np.min(positions[:, 1])
    max_x, max_y = np.max(positions[:, 0]), np.max(positions[:, 1])
    grid_x, grid_y = np.mgrid[min_x:max_x:grid_size*1j, min_y:max_y:grid_size*1j]
    channels = ['x-velocity', 'y-velocity']
    rmse_values, mae_values = [], []
    target_color_ranges = []
    for i in range(2):
        target_values = single_graph.y.cpu()[:, i].numpy() * 7.035423
        color_range = (np.min(target_values), np.max(target_values))
        target_color_ranges.append(color_range)
        rmse_x, rmse_y = rmse_per_component(predicted_features[:, :2]*7.035423, single_graph.y[:, :2]*7.035423)
        mae_x, mae_y = mae_per_component(predicted_features[:, :2]*7.035423, single_graph.y[:, :2]*7.035423)
        rmse_values.append(rmse_x)
        rmse_values.append(rmse_y)
        mae_values.append(mae_x)
        mae_values.append(mae_y)
        print(f"Component {'x-velocity' if i == 0 else 'y-velocity'}: RMSE = {rmse_x if i == 0 else rmse_y}, MAE = {mae_x if i == 0 else mae_y}")
    fig = plt.figure(figsize=(30, 10))
    gs_main = gridspec.GridSpec(2, 10, width_ratios=[1, 0.05, 1, 0.05, 1, 0.05, 1, 0.05, 1, 0.05], wspace=0.1, hspace=0.1)
    for i in range(2):
        for j in range(4):
            plot_col_index = j * 2
            ax = fig.add_subplot(gs_main[i, plot_col_index])
            if j < 2:
                values = single_graph.x.cpu()[:, i].numpy() * 7.035423 if j == 0 else output_graph.x.cpu()[:, i].numpy() * 7.035423
                vmin, vmax = target_color_ranges[i]
            elif j == 2:
                values = single_graph.y.cpu()[:, i].numpy() * 7.035423
                vmin, vmax = target_color_ranges[i]
            elif j == 3:
                values = output_graph.x.cpu()[:, i].numpy() - single_graph.y.cpu()[:, i].numpy()
                vmin, vmax = np.min(values), np.max(values)
            grid_values = griddata(positions, values, (grid_x, grid_y), method='nearest')
            cmap = 'jet' if j < 3 else 'RdBu_r'
            im = ax.imshow(grid_values.T[::-1], extent=(min_x, max_x, min_y, max_y), origin='lower', cmap=cmap, vmin=vmin, vmax=vmax)
            ax.set_aspect('equal', adjustable='box')
            ax.set_title(r'${}$ {} m/s'.format(["Input", "Output", "Target", "Difference"][j], channels[i]))
            ax.set_xticks([])
            ax.set_yticks([])
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="4%", pad=0.05)
            plt.colorbar(im, cax=cax)
    filename = '../results/{}_plot.png'.format(os.path.basename(CHECKPOINT_PATH).split('.')[0])
    fig.savefig(filename, dpi=600, bbox_inches='tight', pad_inches=0.2)
    img = Image.open(filename)
    img.show()
    return rmse_values, mae_values

if __name__ == "__main__":
    label_file  = f'../dataset_graph_full/training_FP/test_graphs_98/cyc11_CAD660_Y7_Z0_X0_label.pt'
    input_file  = f'../dataset_graph_full/training_FP/test_input_graphs_98/cyc11_CAD660_Y7_Z0_X0_input.pt'
    run_GCN(input_file, label_file)
