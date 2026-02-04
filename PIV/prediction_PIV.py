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

from models import GAT_98_8_SkipConnections

MISSING_PERCENTAGE = 90
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CHECKPOINT_PATH = (
    "../trained_models_FP_full/FLUID_skip_GAT_8_98_epochs_20_lr_0.0001_batch_1.pth.tar"
)


def rmse_per_node(pred, target):
    """Compute root mean squared error per node."""
    print(pred.shape, target.shape)
    return torch.sqrt(torch.mean((pred - target) ** 2))


def get_adj_list(edge_index, num_nodes):
    """Convert edge_index to adjacency list representation."""
    adj_list = [[] for _ in range(num_nodes)]
    for i, j in edge_index.t().tolist():
        adj_list[i].append(j)
        adj_list[j].append(i)
    return adj_list


def diffuse_graph_signal(edge_index, x, num_iterations=0, alpha=0.2):
    """Perform heat-based graph signal diffusion."""
    num_nodes = x.size(0)
    adj_list = get_adj_list(edge_index, num_nodes)

    for _ in range(num_iterations):
        diffusion_term = torch.zeros_like(x)
        for node, neighbors in enumerate(adj_list):
            diffusion_term[node] = x[node] - torch.mean(x[neighbors], dim=0)

        x = x - alpha * diffusion_term

    return x


def print_graph_info(graph):
    """Print information about graph structure."""
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
    print(
        f"Percentage of the first three features set to zero: {percentage_zero.item():.2f}%"
    )


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
    state_dict = checkpoint["state_dict"]

    new_state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}

    model.load_state_dict(new_state_dict)


def rmse(pred, target):
    """Compute root mean squared error."""
    return torch.sqrt(torch.mean((pred - target) ** 2))


def mae(pred, target):
    """Compute mean absolute error."""
    return torch.mean(torch.abs(pred - target))


def run_GCN(input_file, label_file):
    """Run GCN prediction on input graph."""
    test_dataset = CustomDataset([input_file], [label_file])
    single_graph = test_dataset[0]

    print("Input Graph:")
    print_graph_info(single_graph)

    label_graph = torch.load(label_file)
    print("Label Graph:")
    print_graph_info(label_graph)

    positions = single_graph.x[:, -2:].numpy()

    print(f"Min x position: {np.min(positions[:, 0])}")
    print(f"Max x position: {np.max(positions[:, 0])}")
    print(f"Min y position: {np.min(positions[:, 1])}")
    print(f"Max y position: {np.max(positions[:, 1])}")

    model = GAT_98_8_SkipConnections()
    model.to(DEVICE)

    load_checkpoint(model, CHECKPOINT_PATH)

    model.eval()

    single_graph.x = single_graph.x.to(DEVICE)
    single_graph.edge_index = single_graph.edge_index.to(DEVICE)
    single_graph.y = single_graph.y.to(DEVICE)

    with torch.no_grad():
        out = model(single_graph)

    indicator = single_graph.x[:, 5] == 1

    corrected_output = out.clone()
    corrected_output[~indicator, :2] = single_graph.x[~indicator, :2]

    diffused_corrected_output = corrected_output.clone()
    diffused_corrected_output[:, :2] = diffuse_graph_signal(
        single_graph.edge_index, corrected_output[:, :2].cpu()
    )

    grid_size = 40

    min_x, min_y = np.min(positions[:, 0]), np.min(positions[:, 1])
    max_x, max_y = np.max(positions[:, 0]), np.max(positions[:, 1])

    grid_x, grid_y = np.mgrid[
        min_x : max_x : grid_size * 1j, min_y : max_y : grid_size * 1j
    ]

    def rmse(pred, target):
        return torch.sqrt(torch.mean((pred - target) ** 2))

    fig, axs = plt.subplots(4, 2, figsize=(10, 10))
    fig.subplots_adjust(hspace=0.5, wspace=0.5)

    def mae(pred, target):
        return torch.mean(torch.abs(pred - target))

    diff_min = np.inf
    diff_max = -np.inf

    channels = ["x-velocity", "y-velocity"]

    for i in range(2):
        scaled_output_values = diffused_corrected_output.cpu()[:, i] * 7.035423
        scaled_target_values = single_graph.y.cpu()[:, i] * 7.035423

        node_wise_rmse = rmse_per_node(scaled_output_values, scaled_target_values)

        print(f"Node-wise RMSE for channel {i}: {node_wise_rmse}")

        input_values = single_graph.x.cpu()[:, i].numpy() * 7.035423
        output_values = diffused_corrected_output.cpu()[:, i].numpy() * 7.035423
        target_values = single_graph.y.cpu()[:, i].numpy() * 7.035423

        grid_input_values = griddata(
            positions, input_values, (grid_x, grid_y), method="nearest"
        )
        grid_output_values = griddata(
            positions, output_values, (grid_x, grid_y), method="nearest"
        )
        grid_target_values = griddata(
            positions, target_values, (grid_x, grid_y), method="nearest"
        )

        vmin, vmax = target_values.min(), target_values.max()

        im = axs[0, i].imshow(
            grid_input_values.T[::-1],
            extent=(min_x, max_x, min_y, max_y),
            origin="lower",
            cmap="jet",
            vmin=vmin,
            vmax=vmax,
        )
        axs[0, i].set_title(f"Input {channels[i]}")
        axs[0, i].set_xticks([])
        axs[0, i].set_yticks([])
        cbar1 = fig.colorbar(im, ax=axs[0, i])
        cbar1.ax.tick_params(labelsize=8)

        im = axs[1, i].imshow(
            grid_output_values.T[::-1],
            extent=(min_x, max_x, min_y, max_y),
            origin="lower",
            cmap="jet",
            vmin=vmin,
            vmax=vmax,
        )
        axs[1, i].set_title(f"Output {channels[i]}")
        axs[1, i].set_xticks([])
        axs[1, i].set_yticks([])
        cbar2 = fig.colorbar(im, ax=axs[1, i])
        cbar2.ax.tick_params(labelsize=8)

        im = axs[2, i].imshow(
            grid_target_values.T[::-1],
            extent=(min_x, max_x, min_y, max_y),
            origin="lower",
            cmap="jet",
            vmin=vmin,
            vmax=vmax,
        )
        axs[2, i].set_title(f"Target {channels[i]}")
        axs[2, i].set_xticks([])
        axs[2, i].set_yticks([])
        cbar3 = fig.colorbar(im, ax=axs[2, i])
        cbar3.ax.tick_params(labelsize=8)

        diff_min = min(diff_min, np.min(grid_output_values - grid_target_values))
        diff_max = max(diff_max, np.max(grid_output_values - grid_target_values))

    for i in range(2):
        scaled_output_values = diffused_corrected_output.cpu()[:, i].numpy() * 7.035423
        scaled_target_values = single_graph.y.cpu()[:, i].numpy() * 7.035423

        diff_values = scaled_output_values - scaled_target_values

        grid_diff_values = griddata(
            positions, diff_values, (grid_x, grid_y), method="nearest"
        )

        cmap = "seismic"
        im = axs[3, i].imshow(
            grid_diff_values.T[::-1],
            extent=(min_x, max_x, min_y, max_y),
            origin="lower",
            cmap=cmap,
            vmin=diff_min,
            vmax=diff_max,
        )
        axs[3, i].set_title(f"Difference {channels[i]}")
        axs[3, i].set_xticks([])
        axs[3, i].set_yticks([])

        cbar = fig.colorbar(im, ax=axs[3, i], orientation="vertical")
        cbar.set_label("Difference magnitude", size=10)
        cbar.ax.tick_params(labelsize=8)

    plt.tight_layout()
    fig.savefig(
        "../results/{}_plot.png".format(
            os.path.basename(CHECKPOINT_PATH).split(".")[0]
        ),
        dpi=600,
    )
    plt.show()


if __name__ == "__main__":
    input_file = f"../PIV_data/test_graphs/test_input_graphs_{MISSING_PERCENTAGE}/PIV_cyc_10_CAD_625_input.pt"
    label_file = f"../PIV_data/test_graphs/test_graphs_{MISSING_PERCENTAGE}/PIV_cyc_10_CAD_625_label.pt"

    run_GCN(input_file, label_file)
