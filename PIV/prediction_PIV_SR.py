import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
from torch.utils.data import Dataset as TorchDataset
from torch_geometric.utils import to_networkx
import networkx as nx
import h5py
from scipy.ndimage import zoom
from models import GAT_98_8_SkipConnections

MISSING_PERCENTAGE = 70
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CHECKPOINT_PATH = (
    "../trained_models_FP_full/FLUID_skip_GAT_8_98_epochs_50_lr_0.0001_batch_1.pth.tar"
)
UREF = 7.035423


def compare_positions(graph1, graph2, description):
    positions1 = graph1.x[:, -2:].cpu().numpy()
    positions2 = graph2.x[:, -2:].cpu().numpy()

    if positions1.shape != positions2.shape:
        print(f"Mismatch in the number of nodes between {description}.")
    else:
        position_difference = np.abs(positions1 - positions2)
        max_position_difference = np.max(position_difference)
        print(
            f"Maximum difference in node positions between {description}: {max_position_difference}"
        )


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
        adj_list[j].append(i)
    return adj_list


def diffuse_graph_signal(edge_index, x, num_iterations=0, alpha=0.2):
    """
    Perform heat-based graph signal diffusion using adjacency list.

    Parameters:
    - edge_index (LongTensor): The edge indices.
    - x (Tensor): Node features to be diffused.
    - num_iterations (int): Number of diffusion iterations.
    - alpha (float): Diffusion coefficient.

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
    state_dict = checkpoint["state_dict"]

    new_state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}

    model.load_state_dict(new_state_dict)


def rmse(pred, target):
    """Computes root mean squared error"""
    return torch.sqrt(torch.mean((pred - target) ** 2))


def mae(pred, target):
    """Computes mean absolute error"""
    return torch.mean(torch.abs(pred - target))


def run_GCN(input_file, label_file, indices_rp_file, mat_file, apply_engine_mask):
    test_dataset = CustomDataset([input_file], [label_file])

    indices_rp = np.load(indices_rp_file)

    single_graph = test_dataset[0]

    print("Input Graph:")
    print_graph_info(single_graph)

    label_graph = torch.load(label_file)
    print("Label Graph:")
    print_graph_info(label_graph)

    positions_input = single_graph.x[:, -2:].numpy()
    positions_target = single_graph.y[:, -2:].numpy()

    print(f"Min x position: {np.min(positions_input[:, 0])}")
    print(f"Max x position: {np.max(positions_input[:, 0])}")
    print(f"Min y position: {np.min(positions_input[:, 1])}")
    print(f"Max y position: {np.max(positions_input[:, 1])}")

    model = GAT_98_8_SkipConnections()
    model.to(DEVICE)
    load_checkpoint(model, CHECKPOINT_PATH)
    model.eval()

    single_graph.to(DEVICE)

    original_positions = single_graph.x[:, -2:].clone()

    with torch.no_grad():
        predicted_features = model(single_graph)

    predicted_features[:, -2:] = original_positions.to(DEVICE)

    output_graph = single_graph.clone()
    output_graph.x = predicted_features

    print("\nComparing node positions between output and input graphs:")
    compare_positions(output_graph, single_graph, "output and input graphs")

    diffused_corrected_output = predicted_features.clone()

    grid_size_input = int(np.sqrt(len(single_graph.x)))
    grid_size_target = int(np.sqrt(len(single_graph.y)))

    min_x, min_y = np.min(positions_input[:, 0]), np.min(positions_input[:, 1])
    max_x, max_y = np.max(positions_input[:, 0]), np.max(positions_input[:, 1])

    grid_x_input, grid_y_input = np.mgrid[
        min_x : max_x : grid_size_input * 1j, min_y : max_y : grid_size_input * 1j
    ]
    grid_x_target, grid_y_target = np.mgrid[
        min_x : max_x : grid_size_target * 1j, min_y : max_y : grid_size_target * 1j
    ]

    fig, axs = plt.subplots(3, 2, figsize=(10, 10))
    fig.subplots_adjust(hspace=0.5, wspace=0.5)

    channels = ["x-velocity", "y-velocity"]

    for i in range(2):
        input_values = single_graph.x.cpu()[:, i].numpy() * UREF
        output_values = diffused_corrected_output.cpu()[:, i].numpy() * UREF
        target_values = single_graph.y.cpu()[:, i].numpy() * UREF

        grid_input_values = griddata(
            positions_input,
            input_values,
            (grid_x_input, grid_y_input),
            method="nearest",
        )
        grid_output_values = griddata(
            positions_input,
            output_values,
            (grid_x_input, grid_y_input),
            method="nearest",
        )
        grid_target_values = griddata(
            positions_target,
            target_values,
            (grid_x_target, grid_y_target),
            method="nearest",
        )

        if apply_engine_mask:
            mat_data = h5py.File(mat_file, "r")
            mask = mat_data["Vel"]["mask"]
            mask_cad = mask[53, :]

            scaling_factor_input = (
                grid_input_values.shape[0] / mask_cad.shape[0],
                grid_input_values.shape[1] / mask_cad.shape[1],
            )
            scaling_factor_target = (
                grid_target_values.shape[0] / mask_cad.shape[0],
                grid_target_values.shape[1] / mask_cad.shape[1],
            )
            zoom_input_mask_cad = zoom(
                mask_cad, zoom=scaling_factor_input, order=0, mode="nearest"
            )
            zoom_target_mask_cad = zoom(
                mask_cad, zoom=scaling_factor_target, order=0, mode="nearest"
            )

            grid_input_values = np.where(
                zoom_input_mask_cad[:, ::-1] < 0.99, np.nan, grid_input_values
            )
            grid_output_values = np.where(
                zoom_input_mask_cad[:, ::-1] < 0.99, np.nan, grid_output_values
            )
            grid_target_values = np.where(
                zoom_target_mask_cad[:, ::-1] < 0.99, np.nan, grid_target_values
            )

        vmin, vmax = target_values.min(), target_values.max()
        node_wise_rmse = rmse(
            diffused_corrected_output[indices_rp, i] * UREF,
            single_graph.y[indices_rp, i] * UREF,
        )
        print(f"Node-wise RMSE for {channels[i]}: {node_wise_rmse}")
        node_wise_mae = mae(
            diffused_corrected_output[indices_rp, i] * UREF,
            single_graph.y[indices_rp, i] * UREF,
        )
        print(f"Node-wise MAE for {channels[i]}: {node_wise_mae}")

        im = axs[0, i].imshow(
            grid_input_values.T,
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
            grid_output_values.T,
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
            grid_target_values.T,
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

    plt.tight_layout()
    fig.savefig(
        "../results/{}_plot_SR.png".format(
            os.path.basename(CHECKPOINT_PATH).split(".")[0]
        ),
        dpi=600,
    )
    plt.show()


if __name__ == "__main__":
    input_file = f"../PIV_data/test_graphs/test_input_graphs_{MISSING_PERCENTAGE}/PIV_cyc_10_CAD_625_full_input_SR.pt"
    label_file = f"../PIV_data/test_graphs/test_graphs_{MISSING_PERCENTAGE}/PIV_cyc_10_CAD_625_full_label_SR.pt"
    indices_rp_file = f"../PIV_data/labels_npz_inputs_{MISSING_PERCENTAGE}/PIV_cyc_10_CAD_625_full_indices_rp.npy"
    mat_file = "../../piv_data/files/OP-C_181114A005.mat"
    apply_engine_mask = True

    run_GCN(input_file, label_file, indices_rp_file, mat_file, apply_engine_mask)
