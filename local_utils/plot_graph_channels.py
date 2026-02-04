import torch
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import griddata


def calculate_mean_divergence_sparse(edge_index, node_features, edge_distance):
    """
    Calculate the mean divergence for the entire graph using a single edge distance feature.

    Parameters:
    - edge_index (LongTensor): The edge indices of the graph.
    - node_features (Tensor): Node features (velocity components).
    - edge_distance (Tensor): Edge feature representing the distance between nodes.

    Returns:
    - float: Mean divergence of the graph.
    """
    num_nodes = node_features.size(0)
    u, v = node_features[:, 0], node_features[:, 1]

    indices = edge_index
    values = edge_distance
    size = torch.Size([num_nodes, num_nodes])
    edge_distance_matrix = torch.sparse.FloatTensor(indices, values, size)

    sum_distances = torch.sparse.sum(edge_distance_matrix, dim=1).to_dense()

    sum_distances[sum_distances == 0] = 1

    du = (
        torch.sparse.mm(edge_distance_matrix, u.unsqueeze(1)).to_dense().squeeze()
        - u * sum_distances
    )
    dv = (
        torch.sparse.mm(edge_distance_matrix, v.unsqueeze(1)).to_dense().squeeze()
        - v * sum_distances
    )

    du_dx = du / sum_distances
    dv_dy = dv / sum_distances
    divergence = du_dx + dv_dy

    mean_divergence = torch.abs(torch.mean(divergence)).item()

    return mean_divergence


graph = torch.load(
    "../dataset_graph_full/training_FP/test_graphs_98/cyc09_CAD615_Y6_Z1_X1_label.pt"
)

node_features = graph.x
edge_index = graph.edge_index
edge_distance = graph.edge_attr

u_velocity = node_features[:, 0].numpy()
v_velocity = node_features[:, 1].numpy()

velocity_magnitude = np.sqrt(u_velocity**2 + v_velocity**2) * 7.035423

positions = node_features[:, -2:].numpy()

mean_divergence = calculate_mean_divergence_sparse(
    edge_index, node_features, edge_distance
)
print(f"Mean divergence of the graph: {mean_divergence}")

grid_size = 256

min_x, min_y = np.min(positions[:, 0]), np.min(positions[:, 1])
max_x, max_y = np.max(positions[:, 0]), np.max(positions[:, 1])

grid_x, grid_y = np.mgrid[
    min_x : max_x : grid_size * 1j, min_y : max_y : grid_size * 1j
]

grid_velocity_magnitude = griddata(
    positions, velocity_magnitude, (grid_x, grid_y), method="nearest"
)

plt.figure(figsize=(8, 8))
plt.imshow(
    grid_velocity_magnitude.T,
    extent=(min_x, max_x, min_y, max_y),
    origin="lower",
    cmap="jet",
)
plt.colorbar(label="Velocity Magnitude")
plt.title("Interpolated Velocity Magnitude")
plt.xlabel("X Position")
plt.ylabel("Y Position")
plt.tight_layout()
plt.savefig("interpolated_velocity_magnitude.png", dpi=1200)

plt.show()
