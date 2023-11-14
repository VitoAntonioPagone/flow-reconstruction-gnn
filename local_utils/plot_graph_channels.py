import torch
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import griddata

# Load the .pt file
#graph = torch.load("../dataset_graph/training/validation_input_graphs_box_90.0/cyc11_CAD605_Y3_Z1_X1_input.pt")
#graph = torch.load("../dataset_graph_full/training_FP/train_input_graphs_98/cyc04_CAD605_Y13_Z0_X1_input.pt")
graph = torch.load("../PIV_data/test_graphs/test_graphs_90/PIV_cyc_10_CAD_625_label.pt")

# Get node features
node_features = graph.x

# Check the shape of the loaded tensor
print(f"Loaded node features shape: {node_features.shape}")

# Assuming the first 3 features are velocities
velocities = node_features[:, :3].numpy()

# Calculate number of zero-velocity nodes and their percentage
zero_velocity_nodes = np.all(velocities == 0, axis=1)
percentage_zero_velocity_nodes = np.mean(zero_velocity_nodes) * 100

# Print the results
print(f"Percentage of nodes with zero velocity: {percentage_zero_velocity_nodes:.2f}%")

# Assuming that the positions are the last 2 features in the feature vector
positions = node_features[:, -2:].numpy()

# Print the features for 5 random nodes
random_indices = np.random.choice(node_features.shape[0], 5, replace=False)
print("Features for 5 random nodes:")
for idx in random_indices:
    print(f"Node {idx}: {node_features[idx].numpy()}")

# Check if the graph has edges
if hasattr(graph, 'edge_index'):
    num_edges = graph.edge_index.shape[1]
    print(f"Number of edges: {num_edges}")
else:
    print("No edges found in the graph.")

# Check if the graph has edge attributes (weights)
if hasattr(graph, 'edge_attr'):
    edge_weights = graph.edge_attr.numpy()
    num_edge_weights = edge_weights.shape[0]  # Assuming edge weights are a 1D array
    print(f"Number of edge weights: {num_edge_weights}")

    # Check if the number of edge weights is equal to the number of edges
    if num_edge_weights == num_edges:
        print("The number of edge weights is equal to the number of edges.")
    else:
        print("The number of edge weights is NOT equal to the number of edges. There might be an issue.")

    # The rest of your code for processing and plotting...
else:
    print("No edge weights found in the graph.")

# Print the dimensionality of the graph
if hasattr(graph, 'edge_index'):
    num_edges = graph.edge_index.shape[1]
    print(f"Dimensionality of the graph: Nodes={node_features.shape[0]}, Edges={num_edges}")
else:
    print(f"Dimensionality of the graph: Nodes={node_features.shape[0]}")

# Define grid size
grid_size = 256  # Increased for a smoother plot

# Get minimum and maximum position values
min_x, min_y = np.min(positions[:, 0]), np.min(positions[:, 1])
max_x, max_y = np.max(positions[:, 0]), np.max(positions[:, 1])

# Create the grid
grid_x, grid_y = np.mgrid[min_x:max_x:grid_size*1j, min_y:max_y:grid_size*1j]

fig, axs = plt.subplots(1, 3, figsize=(18, 6))  # 1 row for 3 channels (velocities)

for i in range(3):  # iterate over velocity channels
    velocities = node_features[:, i].numpy()  # Retrieve the velocity for the current channel

    # Interpolate the values onto the regular grid
    grid_velocities = griddata(positions, velocities, (grid_x, grid_y), method='nearest')

    # Plotting the grid
    im = axs[i].imshow(grid_velocities.T, extent=(min_x, max_x, min_y, max_y), origin='lower', cmap='jet')
    axs[i].set_xlim(min_x, max_x)
    axs[i].set_ylim(min_y, max_y)
    axs[i].set_title(f'Interpolated Velocity Channel {i+1}')
    fig.colorbar(im, ax=axs[i], orientation='vertical')
# Check if the graph has edge attributes (weights)
if hasattr(graph, 'edge_attr'):
    edge_weights = graph.edge_attr.numpy()
    print(f"Loaded edge weights shape: {edge_weights.shape}")

    # Calculate and print statistics about edge weights
    min_edge_weight = np.min(edge_weights)
    max_edge_weight = np.max(edge_weights)
    mean_edge_weight = np.mean(edge_weights)
    std_edge_weight = np.std(edge_weights)

    print(f"Edge Weight Statistics:\n"
          f"Min: {min_edge_weight:.4f}\n"
          f"Max: {max_edge_weight:.4f}\n"
          f"Mean: {mean_edge_weight:.4f}\n"
          f"Std: {std_edge_weight:.4f}")

    # Plot a histogram of edge weights
    plt.figure(figsize=(10, 6))
    plt.hist(edge_weights, bins=50, color='blue', alpha=0.7)
    plt.title('Histogram of Edge Weights')
    plt.xlabel('Edge Weight')
    plt.ylabel('Frequency')
    plt.grid(True)
    plt.show()
else:
    print("No edge weights found in the graph.")
plt.tight_layout()
plt.show()
