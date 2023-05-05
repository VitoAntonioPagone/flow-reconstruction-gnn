import numpy as np
import networkx as nx
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt
import glob

file_pattern = "flow_reconstruction/dataset_graph/original_data/npz_data/train/*.npz"
files = glob.glob(file_pattern)

x_list, y_list, x_velocity_list, y_velocity_list, z_velocity_list, temperature_list = [], [], [], [], [], []

for file in files:
    data = np.load(file)
    x_list.append(data["x"])
    y_list.append(data["y"])
    x_velocity_list.append(data["x_velocity"])
    y_velocity_list.append(data["y_velocity"])
    z_velocity_list.append(data["z_velocity"])
    temperature_list.append(data["temperature"])

x = np.concatenate(x_list)
y = np.concatenate(y_list)
x_velocity = np.concatenate(x_velocity_list)
y_velocity = np.concatenate(y_velocity_list)
z_velocity = np.concatenate(z_velocity_list)
temperature = np.concatenate(temperature_list)
coordinates = np.column_stack((x, y))
features = np.column_stack((x_velocity, y_velocity, z_velocity, temperature))

n_points = len(coordinates)
subset_size = int(0.01 * n_points)  
subset_indices = np.random.choice(n_points, size=subset_size, replace=False)

coordinates_subset = coordinates[subset_indices]
features_subset = features[subset_indices]

distance_matrix = cdist(coordinates_subset, coordinates_subset)

distance_threshold = 0.0125  

#  graph
G = nx.Graph()

# add nodes to the graph
for i in range(len(coordinates_subset)):
    G.add_node(i, features=features_subset[i])

# add edges to the graph based on the distance threshold
for i in range(len(coordinates_subset)):
    for j in range(i + 1, len(coordinates_subset)):
        if distance_matrix[i, j] <= distance_threshold:
            G.add_edge(i, j, weight=distance_matrix[i, j])

# plot 
pos = {i: coordinates_subset[i] for i in range(len(coordinates_subset))}
nx.draw(G, pos, node_size=20, edge_color='gray', node_color='blue', with_labels=False)
plt.show()