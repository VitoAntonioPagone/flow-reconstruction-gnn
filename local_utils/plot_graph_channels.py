import torch
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import griddata

import torch
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import griddata

import torch

def calculate_mean_divergence_sparse(edge_index, node_features, edge_distance):
    """
    Calculate the mean divergence for the entire graph using a single edge distance feature.
    This implementation uses sparse tensor operations for efficiency.

    Parameters:
    - edge_index (LongTensor): The edge indices of the graph.
    - node_features (Tensor): Node features (velocity components).
    - edge_distance (Tensor): Edge feature representing the distance between nodes.
    
    Returns:
    - float: Mean divergence of the graph.
    """
    num_nodes = node_features.size(0)
    u, v = node_features[:, 0], node_features[:, 1]  # Velocity components

    # Create a sparse tensor for edge distances
    indices = edge_index
    values = edge_distance
    size = torch.Size([num_nodes, num_nodes])
    edge_distance_matrix = torch.sparse.FloatTensor(indices, values, size)

    # Use sparse matrix multiplication to compute the sum of distances for each node
    sum_distances = torch.sparse.sum(edge_distance_matrix, dim=1).to_dense()

    # Avoid division by zero
    sum_distances[sum_distances == 0] = 1

    # Compute derivatives using sparse matrix operations
    du = torch.sparse.mm(edge_distance_matrix, u.unsqueeze(1)).to_dense().squeeze() - u * sum_distances
    dv = torch.sparse.mm(edge_distance_matrix, v.unsqueeze(1)).to_dense().squeeze() - v * sum_distances

    # Calculate divergence at each node and then compute the mean
    du_dx = du / sum_distances
    dv_dy = dv / sum_distances
    divergence = du_dx + dv_dy

    mean_divergence = torch.abs(torch.mean(divergence)).item()

    return mean_divergence

# Load the .pt file
graph = torch.load("../dataset_graph_full/training_FP/test_graphs_98/cyc09_CAD615_Y6_Z1_X1_label.pt")

# Extract node features and edge index
node_features = graph.x
edge_index = graph.edge_index
edge_distance = graph.edge_attr  # Assuming edge_attr contains the edge distances

# Assuming the first 2 features are the x and y components of the velocity
u_velocity = node_features[:, 0].numpy()  # x component of velocity
v_velocity = node_features[:, 1].numpy()  # y component of velocity

# Calculate velocity magnitude for each node
velocity_magnitude = np.sqrt(u_velocity**2 + v_velocity**2) * 7.035423

# Assuming that the positions are the last 2 features in the feature vector
positions = node_features[:, -2:].numpy()

# Calculate mean divergence using the function
mean_divergence = calculate_mean_divergence_sparse(edge_index, node_features, edge_distance)
print(f"Mean divergence of the graph: {mean_divergence}")

# Define grid size
grid_size = 256  # Adjust the grid size as needed

# Get minimum and maximum position values
min_x, min_y = np.min(positions[:, 0]), np.min(positions[:, 1])
max_x, max_y = np.max(positions[:, 0]), np.max(positions[:, 1])

# Create the grid
grid_x, grid_y = np.mgrid[min_x:max_x:grid_size*1j, min_y:max_y:grid_size*1j]

# Interpolate the velocity magnitude onto the regular grid
grid_velocity_magnitude = griddata(positions, velocity_magnitude, (grid_x, grid_y), method='nearest')

# Plotting the interpolated velocity magnitude
plt.figure(figsize=(8, 8))
plt.imshow(grid_velocity_magnitude.T, extent=(min_x, max_x, min_y, max_y), origin='lower', cmap='jet')
plt.colorbar(label='Velocity Magnitude')
plt.title('Interpolated Velocity Magnitude')
plt.xlabel('X Position')
plt.ylabel('Y Position')
plt.tight_layout()
plt.savefig('interpolated_velocity_magnitude.png', dpi=1200)

plt.show()